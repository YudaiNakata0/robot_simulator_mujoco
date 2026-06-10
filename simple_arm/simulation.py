import time
import mujoco
import mujoco.viewer
import numpy as np

# 1. 作成したXMLモデルの読み込み
model = mujoco.MjModel.from_xml_path('robot.xml')
data = mujoco.MjData(model)

# 2. シミュレーション画面（Viewer）を立ち上げる
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 初期姿勢を少し曲げた状態にする
    data.qpos[0] = 0.2
    data.qpos[1] = 0.5
    data.qpos[2] = -0.5
    data.qpos[3] = 0.2
    
    print("シミュレーションを開始します。")
    
    step_count = 0  # 💡【修正】ステップ数カウント用の変数を定義
    
    while viewer.is_running():
        step_start = time.time()

        # 3. 物理演算を1ステップ進める
        mujoco.mj_step(model, data)
        
        # --- ここから課題の核心（ヤコビアンの取得） ---
        # 手先（'tip'）のIDを取得
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip')
        
        # ヤコビアンを格納する空の行列を用意
        jac_p = np.zeros((3, model.nv)) # 並進ヤコビアン (3x4)
        jac_r = np.zeros((3, model.nv)) # 回転ヤコビアン (3x4)
        
        # 現在の関節状態から手先のヤコビアンを計算
        mujoco.mj_jac(model, data, jac_p, jac_r, data.site_xpos[site_id], site_id)
        
        # X-Z平面の2次元なので、並進成分 (jac_p) のうち X（0行目）と Z（2行目）だけを使う
        J = jac_p[[0, 2], :] # 2x4 のタスクヤコビアン
        
        # 💡【修正】自前のカウンターを使って100ステップに1回ヤコビアンをプリント
        if step_count % 100 == 0:
            print(f"--- Time: {data.time:.2f}s ---")
            print("現在のタスクヤコビアン J (2x4):\n", J)
        # --------------------------------------------

        # アニメーションを更新
        viewer.sync()
        
        step_count += 1  # 💡【修正】カウンターをインクリメント

        # 実際の時間経過と同期させるためのウェイト
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)