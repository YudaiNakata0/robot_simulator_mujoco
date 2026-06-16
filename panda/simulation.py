import time
import mujoco
import mujoco.viewer
import numpy as np
import threading

class PandaSimulator:
    def __init__(self):
        self.target_pos = np.array([0.1, 0.0, 0.8])  # 目標位置
        self.target_lock = threading.Lock()
        self.K_p = 2  # 比例ゲイン

    def input_thread(self):
        while True:
            try:
                s = input("target x y z > ")

                vals = [float(v) for v in s.split()]

                if len(vals) != 3:
                    print("入力形式: x y z")
                    continue

                with self.target_lock:
                    self.target_pos[:] = vals

                print("new target =", self.target_pos)

            except Exception as e:
                print(e)

    def main(self):
        threading.Thread(
        target=self.input_thread,
        daemon=True
        ).start()
        # 1. 作成したXMLモデルの読み込み
        model = mujoco.MjModel.from_xml_path('scene.xml')
        data = mujoco.MjData(model)

        # 2. シミュレーション画面（Viewer）を立ち上げる
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            # 初期姿勢
            home_key = model.key("home")

            data.qpos[:] = home_key.qpos

            mujoco.mj_forward(model, data)    
            
            print("シミュレーションを開始します。")
            
            step_count = 0
            
            # サーボの位置取得
            q_des = data.qpos[:7].copy()
            
            while viewer.is_running():
                step_start = time.time()

                # 3. 物理演算を1ステップ進める
                mujoco.mj_step(model, data)
                
                # 手先（'tip'）のIDを取得
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip')
                
                # 現在の手先位置を取得し速度指令を計算
                x = data.site_xpos[site_id].copy()
                with self.target_lock:
                    x_target = self.target_pos.copy()
                error = x_target - x
                v_des = self.K_p * error
                # print(f"Step {step_count}: v_des = {v_des}, error = {error}")
                
                # ヤコビアンを格納する空の行列を用意
                jac_p = np.zeros((3, model.nv))
                jac_r = np.zeros((3, model.nv))
                
                # 現在の関節状態から手先のヤコビアンを計算
                mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)

                # 指を覗いた7自由度を制御
                J_pos = jac_p[:, :7]
                # print(f"Step {step_count}: J_pos = {J_pos}")
                dq = np.linalg.pinv(J_pos) @ v_des
                # print(f"Step {step_count}: dq = {dq}, v_des = {v_des}, error = {error}")
                
                # 関節位置の更新
                q_des += dq * model.opt.timestep
                # print(f"dq {dq}: q_des = {q_des}, duration = {model.opt.timestep}")
                data.ctrl[:7] = q_des

                # アニメーションを更新
                viewer.sync()
                
                step_count += 1

                # 実際の時間経過と同期させるためのウェイト
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    simulator = PandaSimulator()
    simulator.main()