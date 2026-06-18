import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import threading
import tkinter as tk
import tkinter.ttk as ttk
from collections import deque

FR3_DIR = os.path.join(os.path.dirname(__file__), "")

SCENE_XML = """
<mujoco model="fr3 scene">
  <include file="fr3.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>

  <actuator>
    <motor name="mot1" joint="fr3_joint1" gear="1" forcerange="-87 87"/>
    <motor name="mot2" joint="fr3_joint2" gear="1" forcerange="-87 87"/>
    <motor name="mot3" joint="fr3_joint3" gear="1" forcerange="-87 87"/>
    <motor name="mot4" joint="fr3_joint4" gear="1" forcerange="-87 87"/>
    <motor name="mot5" joint="fr3_joint5" gear="1" forcerange="-12 12"/>
    <motor name="mot6" joint="fr3_joint6" gear="1" forcerange="-12 12"/>
    <motor name="mot7" joint="fr3_joint7" gear="1" forcerange="-12 12"/>
  </actuator>
</mujoco>
"""

TMP_XML = os.path.join(FR3_DIR, "_tmp_fr3_circle.xml")
with open(TMP_XML, "w") as f:
    f.write(SCENE_XML)

N_JOINTS = 7

# --- 初期関節角度 [rad]（None にすると MuJoCo のデフォルト姿勢）---
# FR3 の関節範囲:
#   joint1: -2.8973 ~ 2.8973
#   joint2: -1.7628 ~ 1.7628
#   joint3: -2.8973 ~ 2.8973
#   joint4: -3.0718 ~ -0.0698
#   joint5: -2.8973 ~ 2.8973
#   joint6: -0.0175 ~ 3.7525
#   joint7: -2.8973 ~ 2.8973
INIT_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
# INIT_QPOS = None  # ← デフォルト姿勢を使う場合はこちら

# --- 可変パラメータ（実行中に変更可能）---
class Params:
    radius       = 0.1
    omega        = 2 * np.pi * 0.1
    cx, cy, cz   = 0.0, 0.2, 0.6   # 円中心オフセット
    kp           = 100.0            # OSC 位置ゲイン
    ki           = 30.0             # OSC 積分ゲイン（定常誤差除去）
    kd           = 5.0              # OSC 速度ゲイン（減衰）
    i_clamp      = 20.0             # 積分クランプ [m/s²相当]
    null_gain    = 5.0              # ヌルスペース位置ゲイン（スカラー）
    dominance    = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    jvg          = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    damping      = 1e-3             # Λ 計算時の正則化
    low_gain     = 00.0           # リンクを手先Z以下に抑える下向きゲイン
    axis7_gain   = 0.0             # 関節7軸のZ成分を正に保つゲイン（0で無効）
    # --- 感情パラメータ ---
    jerk_amp     = 0.0              # ジャーク注入の振幅 [m/s³] (P-で大きくなる)
    jerk_freq    = 2.0              # ジャーク注入の周波数 [Hz]
    emotion      = "neutral"        # 現在の感情名
    reset_integral = False

P = Params()

# --- PAD 8感情プリセット ---
# 軌道（omega, radius）は変えない。ヌルスペースとジャークだけで感情を表現。
#
# P（快）    : jerk_amp 小→滑らか、大→ぎこちない
# A（覚醒）  : null_freq_scale 大→速い関節揺動、小→ゆっくり
# D（支配）  : null_amp_scale  大→大きな関節揺動、小→小さい
#
# ヌルスペース目標 q_null(t) = q_center + amp * sin(freq_i * t + phi_i)
# amp・freq はランダム初期化、感情でスケーリング

EMOTION_PRESETS = {
    #            P    A    D     jerk_amp  jerk_freq  null_amp_scale  null_freq_scale
    "joy":      dict(jerk_amp=0.0,  jerk_freq=0.0, null_amp=0.4, null_freq=1.5),  # P+A+D+
    "friendly": dict(jerk_amp=0.0,  jerk_freq=0.0, null_amp=0.4, null_freq=1.5),  # P+A+D-
    "relaxed":  dict(jerk_amp=0.0,  jerk_freq=0.0, null_amp=0.3, null_freq=0.4),  # P+A-D+
    "docile":   dict(jerk_amp=0.0,  jerk_freq=0.0, null_amp=0.3, null_freq=0.4),  # P+A-D-
    "angry":    dict(jerk_amp=80.0, jerk_freq=4.0, null_amp=0.4, null_freq=1.5),  # P-A+D+
    "fear":     dict(jerk_amp=60.0, jerk_freq=6.0, null_amp=0.4, null_freq=1.5),  # P-A+D-
    "disgust":  dict(jerk_amp=40.0, jerk_freq=1.5, null_amp=0.3, null_freq=0.4),  # P-A-D+
    "sad":      dict(jerk_amp=10.0, jerk_freq=0.8, null_amp=0.3, null_freq=0.4),  # P-A-D-
}


class NullMotion:
    """
    全関節に共通した感情スタイルのランダム揺動目標を生成する。
    各関節は同じ amp_scale・freq_scale で、位相・基底周波数だけランダム。
    → 「同じ感情」を全関節で表現しつつ、動き方はランダム。
    """
    N_HARMONICS = 3   # 重ね合わせる正弦波の数

    def __init__(self):
        self._regen(amp_scale=0.2, freq_scale=0.5)

    def _regen(self, amp_scale, freq_scale):
        rng = np.random.default_rng()
        # 各関節 × 各倍音ごとにランダム振幅・周波数・位相
        self.amps  = rng.uniform(0.3, 1.0, (N_JOINTS, self.N_HARMONICS))  # 相対振幅
        self.freqs = rng.uniform(0.5, 2.0, (N_JOINTS, self.N_HARMONICS))  # 相対周波数
        self.phases= rng.uniform(0,  2*np.pi,(N_JOINTS,self.N_HARMONICS))
        self.amp_scale  = amp_scale
        self.freq_scale = freq_scale

    def target(self, q_center, t):
        """時刻 t における各関節の目標角度"""
        q = q_center.copy()
        for i in range(N_JOINTS):
            for h in range(self.N_HARMONICS):
                q[i] += (self.amp_scale * self.amps[i, h]
                         * np.sin(2*np.pi * self.freq_scale * self.freqs[i, h] * t
                                  + self.phases[i, h]))
        return q

null_motion = NullMotion()


def apply_emotion(name):
    name = name.lower()
    if name not in EMOTION_PRESETS:
        print(f"  不明な感情。使えるもの: {list(EMOTION_PRESETS.keys())}")
        return
    preset = EMOTION_PRESETS[name]
    P.jerk_amp  = preset["jerk_amp"]
    P.jerk_freq = preset["jerk_freq"]
    P.emotion   = name
    # ランダム揺動のスケールを感情に合わせて再生成
    null_motion._regen(amp_scale=preset["null_amp"], freq_scale=preset["null_freq"])
    print(f"  感情 → {name}  "
          f"(jerk_amp={P.jerk_amp:.1f}, jerk_freq={P.jerk_freq:.1f}Hz, "
          f"null_amp={preset['null_amp']:.2f}, null_freq={preset['null_freq']:.2f})")

# --- JVG 特徴量（スレッド間共有） ---
# 振り幅計算用の時間窓（秒）
G_WINDOW_SEC = 2.0

class JVGState:
    """メインループで更新 → 表示ウィンドウで読み取る"""
    V = np.zeros(N_JOINTS)   # 関節速度 [rad/s]
    A = np.zeros(N_JOINTS)   # 関節加速度 [rad/s²]
    J = np.zeros(N_JOINTS)   # ジャーク [rad/s³]
    G = np.zeros(N_JOINTS)   # 振り幅（時間窓内の max-min）[rad]
    lock = threading.Lock()

jvg_state = JVGState()


def param_window():
    """パラメータ＋JVG特徴量をリアルタイム表示する別ウィンドウ（タブ切り替え）"""
    root = tk.Tk()
    root.title("パラメータ / JVG特徴量")
    root.geometry("900x900")
    root.resizable(False, False)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # ===== タブ1: パラメータ =====
    tab_param = tk.Frame(notebook)
    notebook.add(tab_param, text="  パラメータ  ")

    param_rows = [
        ("--- 感情 ---",  lambda: ""),
        ("Emotion",      lambda: P.emotion),
        ("JerkAmp",      lambda: f"{P.jerk_amp:.1f}"),
        ("JerkFreq",     lambda: f"{P.jerk_freq:.2f} Hz"),
        ("--- 軌道 ---",  lambda: ""),
        ("半径",         lambda: f"{P.radius:.3f} m"),
        ("角速度",       lambda: f"{P.omega:.3f} rad/s"),
        ("中心 X",       lambda: f"{P.cx:.3f} m"),
        ("中心 Y",       lambda: f"{P.cy:.3f} m"),
        ("中心 Z",       lambda: f"{P.cz:.3f} m"),
        ("--- OSC ---",  lambda: ""),
        ("KP",           lambda: f"{P.kp:.2f}"),
        ("KI",           lambda: f"{P.ki:.2f}"),
        ("KD",           lambda: f"{P.kd:.2f}"),
        ("I_CLAMP",      lambda: f"{P.i_clamp:.2f}"),
        ("--- Null ---", lambda: ""),
        ("NullGain",     lambda: f"{P.null_gain:.3f}"),
        ("LowGain",      lambda: f"{P.low_gain:.1f}"),
        ("Axis7Gain",    lambda: f"{P.axis7_gain:.1f}"),
        ("Dominance",    lambda: ", ".join(f"{v:.1f}" for v in P.dominance)),
        ("JVG(速度減衰)",lambda: ", ".join(f"{v:.1f}" for v in P.jvg)),
        ("Damping",      lambda: f"{P.damping:.2e}"),
    ]
    param_vars = {}
    for i, (name, fn) in enumerate(param_rows):
        tk.Label(tab_param, text=name, width=14, anchor="w",
                 font=("Consolas", 18, "bold" if "---" in name else "normal"),
                 fg="gray" if "---" in name else "black").grid(row=i, column=0, padx=16, pady=3)
        var = tk.StringVar()
        val_width = 42 if name in ("Dominance", "JVG(速度減衰)") else 22
        tk.Label(tab_param, textvariable=var, width=val_width, anchor="w",
                 font=("Consolas", 18)).grid(row=i, column=1, padx=8, pady=3)
        param_vars[name] = (var, fn)

    # ===== タブ2: JVG特徴量 =====
    tab_jvg = tk.Frame(notebook)
    notebook.add(tab_jvg, text="  JVG特徴量  ")

    FONT_H = ("Consolas", 16, "bold")
    FONT_V = ("Consolas", 16)
    joint_names = [f"J{i+1}" for i in range(N_JOINTS)]

    # ヘッダ
    for col, label in enumerate(["関節", "速度V [rad/s]", "ジャークJ [rad/s³]", f"振り幅G [{G_WINDOW_SEC}s, rad]"]):
        tk.Label(tab_jvg, text=label, font=FONT_H, anchor="center",
                 relief="groove", width=18).grid(row=0, column=col, padx=4, pady=6, sticky="ew")

    jvg_vars = {"V": [], "J": [], "G": []}
    for i in range(N_JOINTS):
        tk.Label(tab_jvg, text=joint_names[i], font=FONT_V, width=6,
                 anchor="center").grid(row=i+1, column=0, padx=4, pady=4)
        for col_idx, key in enumerate(["V", "J", "G"]):
            var = tk.StringVar(value="---")
            tk.Label(tab_jvg, textvariable=var, font=FONT_V, width=18,
                     anchor="center", relief="sunken").grid(row=i+1, column=col_idx+1, padx=4, pady=4)
            jvg_vars[key].append(var)

    def update():
        # パラメータ更新
        for name, (var, fn) in param_vars.items():
            var.set(fn())
        # JVG更新
        with jvg_state.lock:
            V = jvg_state.V.copy()
            J = jvg_state.J.copy()
            G = jvg_state.G.copy()
        for i in range(N_JOINTS):
            jvg_vars["V"][i].set(f"{V[i]:+.4f}")
            jvg_vars["J"][i].set(f"{J[i]:+.4f}")
            jvg_vars["G"][i].set(f"{G[i]:.4f}")
        root.after(100, update)

    update()
    root.mainloop()


def command_thread():
    """ターミナルからコマンドを受け付けるスレッド"""
    print("\n=== コマンド一覧（OSC） ===")
    print("  r <値>         : 半径")
    print("  cx/cy/cz <値>  : 円中心オフセット")
    print("  kp <値>        : OSC 位置ゲイン")
    print("  ki <値>        : OSC 積分ゲイン")
    print("  kd <値>        : OSC 速度ゲイン")
    print("  iclamp <値>    : 積分クランプ")
    print("  ng <値>          : ヌルスペース位置ゲイン")
    print("  lg <値>          : リンク押し下げゲイン（手先Z以下に抑える）")
    print("  a7g <値>         : 関節7軸Z成分を正に保つゲイン（0で無効）")
    print("  dom <軸1-7> <値> : Dominance 位置重み（例: dom 4 2.0）")
    print("  jvg <軸1-7> <値> : Joint Velocity Gain 速度減衰（例: jvg 3 3.0）")
    print("  damp <値>        : 正則化パラメータ")
    print("  --- 感情制御 ---")
    print("  emotion <名前>   : 感情プリセット切り替え")
    print("    joy / friendly / relaxed / docile")
    print("    angry / fear / disgust / sad")
    print("  ja <値>          : ジャーク注入振幅")
    print("  jf <値>          : ジャーク注入周波数 [Hz]")
    print("  show             : 現在の値を表示")
    print("==========================\n")
    while True:
        try:
            cmd = input().strip().split()
            if not cmd:
                continue
            key = cmd[0].lower()
            if   key == "r"    and len(cmd) > 1: P.radius    = float(cmd[1]); print(f"  radius = {P.radius}")
            elif key == "cx"   and len(cmd) > 1: P.cx        = float(cmd[1]); print(f"  cx = {P.cx}")
            elif key == "cy"   and len(cmd) > 1: P.cy        = float(cmd[1]); print(f"  cy = {P.cy}")
            elif key == "cz"   and len(cmd) > 1: P.cz        = float(cmd[1]); print(f"  cz = {P.cz}")
            elif key == "kp"     and len(cmd) > 1: P.kp        = float(cmd[1]); print(f"  kp = {P.kp}")
            elif key == "ki"     and len(cmd) > 1: P.ki        = float(cmd[1]); P.reset_integral = True; print(f"  ki = {P.ki}（積分リセット）")
            elif key == "kd"     and len(cmd) > 1: P.kd        = float(cmd[1]); print(f"  kd = {P.kd}")
            elif key == "iclamp" and len(cmd) > 1: P.i_clamp   = float(cmd[1]); P.reset_integral = True; print(f"  i_clamp = {P.i_clamp}（積分リセット）")
            elif key == "ng"   and len(cmd) > 1: P.null_gain = float(cmd[1]); print(f"  null_gain = {P.null_gain}")
            elif key == "lg"   and len(cmd) > 1: P.low_gain   = float(cmd[1]); print(f"  low_gain = {P.low_gain}")
            elif key == "a7g"  and len(cmd) > 1: P.axis7_gain = float(cmd[1]); print(f"  axis7_gain = {P.axis7_gain}")
            elif key == "damp" and len(cmd) > 1: P.damping   = float(cmd[1]); print(f"  damping = {P.damping}")
            elif key == "dom"  and len(cmd) > 2:
                idx = int(cmd[1]) - 1
                P.dominance[idx] = float(cmd[2])
                print(f"  dominance[{idx+1}] = {P.dominance[idx]}")
            elif key == "jvg"  and len(cmd) > 2:
                idx = int(cmd[1]) - 1
                P.jvg[idx] = float(cmd[2])
                print(f"  jvg[{idx+1}] = {P.jvg[idx]}")
            elif key == "emotion" and len(cmd) > 1:
                apply_emotion(cmd[1])
            elif key == "ja"   and len(cmd) > 1: P.jerk_amp  = float(cmd[1]); print(f"  jerk_amp = {P.jerk_amp}")
            elif key == "jf"   and len(cmd) > 1: P.jerk_freq = float(cmd[1]); print(f"  jerk_freq = {P.jerk_freq}")
            elif key == "show":
                print(f"  [感情] {P.emotion}  jerk_amp={P.jerk_amp}  jerk_freq={P.jerk_freq}")
                print(f"  radius={P.radius}  omega={P.omega:.3f}  cx={P.cx} cy={P.cy} cz={P.cz}")
                print(f"  kp={P.kp}  ki={P.ki}  kd={P.kd}  i_clamp={P.i_clamp}")
                print(f"  null_gain={P.null_gain}  damping={P.damping}")
                print(f"  dominance={P.dominance}")
                print(f"  jvg={P.jvg}")
            else:
                print("  不明なコマンド。'show' で一覧確認")
        except Exception as e:
            print(f"  エラー: {e}")


def get_ee_pos(model, data):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return data.site_xpos[site_id].copy()


def get_ee_vel(model, data):
    """手先速度（線速度）を Jv * dq で計算"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    Jv = jacp[:, :N_JOINTS]
    return Jv @ data.qvel[:N_JOINTS]


def compute_jacobian(model, data):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, :N_JOINTS]


def osc_torques(model, data, x_ddot_des, t):
    """
    OSC トルクを計算する。

    τ = Jᵀ Λ x_ddot_des + τ_bias + Nᵀ τ_null

    ヌルスペース目標は感情に応じたランダム揺動軌道 null_motion.target(q_center, t)
    """
    J = compute_jacobian(model, data)   # (3, N_JOINTS)

    # 完全な質量行列 M を取得 (N_JOINTS x N_JOINTS)
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M = M_full[:N_JOINTS, :N_JOINTS]

    # M⁻¹
    M_inv = np.linalg.inv(M)

    # 操作空間慣性行列 Λ = (J M⁻¹ Jᵀ + λI)⁻¹
    JMinvJT = J @ M_inv @ J.T
    lam = np.linalg.inv(JMinvJT + P.damping * np.eye(3))

    # 主タスクトルク
    tau_task = J.T @ lam @ x_ddot_des

    # コリオリ・重力補償（バイアス）
    tau_bias = data.qfrc_bias[:N_JOINTS]

    # ヌルスペース投影行列 N = I - Jᵀ Λ J M⁻¹
    N_proj = np.eye(N_JOINTS) - J.T @ (lam @ J @ M_inv)

    # ヌルスペースタスク：感情スタイルのランダム揺動軌道を追う（PD制御）
    q_center  = 0.5 * (model.jnt_range[:N_JOINTS, 0] + model.jnt_range[:N_JOINTS, 1])
    q_target  = null_motion.target(q_center, t)
    q_target  = np.clip(q_target,
                        model.jnt_range[:N_JOINTS, 0] + 0.1,
                        model.jnt_range[:N_JOINTS, 1] - 0.1)
    q_err     = q_target - data.qpos[:N_JOINTS]
    dq        = data.qvel[:N_JOINTS]
    tau_null_raw = P.null_gain * q_err - P.jvg * dq

    # --- リンク押し下げ：手先Zより上に出たリンクを下に引く ---
    # ee_z = data.site_xpos[
    #     mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    # ][2]
    # for k in range(1, N_JOINTS + 1):
    #     body_name = f"fr3_link{k}"
    #     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #     if body_id < 0:
    #         continue
    #     z_link = data.xpos[body_id][2]
    #     excess = z_link - ee_z
    #     if excess <= 0.1:
    #         continue
    #     jacp_link = np.zeros((3, model.nv))
    #     jacr_link = np.zeros((3, model.nv))
    #     mujoco.mj_jacBody(model, data, jacp_link, jacr_link, body_id)
    #     Jz = jacp_link[2, :N_JOINTS]
    #     tau_null_raw -= P.low_gain * excess * Jz

    # --- 関節7の回転軸Z成分を正に保つ ---
    # if P.axis7_gain > 0:
    #     link7_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link7")
    #     joint7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fr3_joint7")
    #     R7          = data.xmat[link7_id].reshape(3, 3)
    #     axis7_world = R7 @ model.jnt_axis[joint7_id]
    #     z_comp      = axis7_world[2]
    #     if z_comp < 0:
    #         jacp7 = np.zeros((3, model.nv))
    #         jacr7 = np.zeros((3, model.nv))
    #         mujoco.mj_jacBody(model, data, jacp7, jacr7, link7_id)
    #         ez   = np.array([0.0, 0.0, 1.0])
    #         grad = jacr7[:, :N_JOINTS].T @ np.cross(ez, axis7_world)
    #         tau_null_raw += P.axis7_gain * (-z_comp) * grad

    tau_null  = N_proj @ tau_null_raw

    return tau_task + tau_bias + tau_null


def main():
    model = mujoco.MjModel.from_xml_path(TMP_XML)
    data  = mujoco.MjData(model)

    # 初期姿勢を設定
    if INIT_QPOS is not None:
        data.qpos[:N_JOINTS] = INIT_QPOS
        data.qvel[:N_JOINTS] = 0.0

    mujoco.mj_forward(model, data)

    base_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link0")
    base_pos = data.xpos[base_id].copy()

    # コマンド受付スレッド起動
    threading.Thread(target=command_thread, daemon=True).start()
    # パラメータ表示ウィンドウ起動
    threading.Thread(target=param_window, daemon=True).start()

    N_CIRCLE = 100

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 2.0
        viewer.cam.elevation = -25
        viewer.cam.azimuth   = 135
        try:
            viewer._ui0_width = 150
            viewer._ui1_width = 150
        except AttributeError:
            pass

        t = 0.0
        err_integral = np.zeros(3)
        prev_radius = P.radius
        prev_offset = (P.cx, P.cy, P.cz)

        # JVG 計算用バッファ
        prev_vel  = np.zeros(N_JOINTS)
        prev_acc  = np.zeros(N_JOINTS)
        # 振り幅用リングバッファ（時間窓分のqposを保持）
        G_window_steps = max(1, int(G_WINDOW_SEC / model.opt.timestep))
        q_history = deque(maxlen=G_window_steps)

        center = base_pos + np.array([P.cx, P.cy, P.cz])
        circle_points = np.array([
            center + np.array([
                P.radius * np.cos(2 * np.pi * i / N_CIRCLE),
                P.radius * np.sin(2 * np.pi * i / N_CIRCLE),
                0.0
            ]) for i in range(N_CIRCLE + 1)
        ])

        while viewer.is_running():
            dt = model.opt.timestep

            # 円中心をリアルタイムで計算
            center = base_pos + np.array([P.cx, P.cy, P.cz])

            # 半径や中心が変わったら軌道点列を再計算
            if P.radius != prev_radius or (P.cx, P.cy, P.cz) != prev_offset:
                circle_points = np.array([
                    center + np.array([
                        P.radius * np.cos(2 * np.pi * i / N_CIRCLE),
                        P.radius * np.sin(2 * np.pi * i / N_CIRCLE),
                        0.0
                    ]) for i in range(N_CIRCLE + 1)
                ])
                prev_radius = P.radius
                prev_offset = (P.cx, P.cy, P.cz)
                err_integral[:] = 0

            # 積分リセットフラグ
            if P.reset_integral:
                err_integral[:] = 0
                P.reset_integral = False

            # 目標位置・速度・加速度（円軌道）
            pos_target = center + np.array([
                P.radius * np.cos(P.omega * t),
                P.radius * np.sin(P.omega * t),
                0.0
            ])
            vel_target = np.array([
                -P.radius * P.omega * np.sin(P.omega * t),
                 P.radius * P.omega * np.cos(P.omega * t),
                 0.0
            ])
            acc_target = np.array([
                -P.radius * P.omega**2 * np.cos(P.omega * t),
                -P.radius * P.omega**2 * np.sin(P.omega * t),
                 0.0
            ])

            # 現在の手先位置・速度
            pos_cur = get_ee_pos(model, data)
            vel_cur = get_ee_vel(model, data)

            # OSC 目標加速度（PID in task space）
            pos_err      = pos_target - pos_cur
            vel_err      = vel_target - vel_cur
            err_integral += pos_err * dt
            err_integral  = np.clip(err_integral, -P.i_clamp, P.i_clamp)
            x_ddot_des   = acc_target + P.kp * pos_err + P.ki * err_integral + P.kd * vel_err

            # ジャーク注入（P-感情：ぎこちなさの表現）
            # 方向はランダム単位ベクトル、周波数に同期したパルスで注入
            if P.jerk_amp > 0.0 and P.jerk_freq > 0.0:
                phase_jerk = (t * P.jerk_freq) % 1.0
                if phase_jerk < dt * P.jerk_freq:          # 1周期に1回パルス
                    rng_vec = np.random.randn(3)
                    rng_vec /= np.linalg.norm(rng_vec) + 1e-9
                    x_ddot_des += P.jerk_amp * rng_vec

            # トルク計算・適用
            tau = osc_torques(model, data, x_ddot_des, t)
            data.ctrl[:N_JOINTS] = tau

            mujoco.mj_step(model, data)

            # ===== JVG 特徴量の計算 =====
            cur_vel = data.qvel[:N_JOINTS].copy()
            cur_acc = (cur_vel - prev_vel) / dt
            cur_jerk = (cur_acc - prev_acc) / dt
            q_history.append(data.qpos[:N_JOINTS].copy())
            q_arr = np.array(q_history)
            cur_G = q_arr.max(axis=0) - q_arr.min(axis=0)

            with jvg_state.lock:
                jvg_state.V = cur_vel
                jvg_state.A = cur_acc
                jvg_state.J = cur_jerk
                jvg_state.G = cur_G

            prev_vel = cur_vel
            prev_acc = cur_acc

            # 目標軌道を線で描画
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                for i in range(len(circle_points) - 1):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array([1.0, 0.3, 0.0, 1.0], dtype=np.float32)
                    )
                    mujoco.mjv_connector(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        3.0,
                        circle_points[i],
                        circle_points[i + 1]
                    )
                    viewer.user_scn.ngeom += 1

            viewer.sync()

            t += dt
            time.sleep(dt)


if __name__ == "__main__":
    main()
