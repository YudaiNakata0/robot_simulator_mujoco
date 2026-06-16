import time
import mujoco
import mujoco.viewer
import numpy as np
import threading
import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray


class PandaSimulator:
    def __init__(self):
        self.target_pos = np.array([0.1, 0.0, 0.8])  # 目標位置
        self.target_lock = threading.Lock()
        self.K_p = 2  # 比例ゲイン
        self.setup_ros()
        
    def setup_ros(self):
        rospy.init_node('panda_simulator', anonymous=True)
        rospy.Subscriber('/target_position', Vector3, self.callback_target_position)
        self.position_pub = rospy.Publisher('/current_position', Vector3, queue_size=10)
        self.joint_pub = rospy.Publisher('/joint_states', Float64MultiArray, queue_size=10)

    def callback_target_position(self, msg):
        with self.target_lock:
            self.target_pos[:] = [msg.x, msg.y, msg.z]

    def main(self):
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
            
            while viewer.is_running() and not rospy.is_shutdown():
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
                
                # ヤコビアンを格納する空の行列を用意
                jac_p = np.zeros((3, model.nv))
                jac_r = np.zeros((3, model.nv))
                
                # 現在の関節状態から手先のヤコビアンを計算
                mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)

                # 指を覗いた7自由度を制御
                J_pos = jac_p[:, :7]
                dq = np.linalg.pinv(J_pos) @ v_des
                
                # 関節位置の更新
                q_des += dq * model.opt.timestep
                data.ctrl[:7] = q_des
                
                # 状態の配信
                msg_position = Vector3()
                msg_position.x = x[0]
                msg_position.y = x[1]
                msg_position.z = x[2]
                self.position_pub.publish(msg_position)

                msg_joints = Float64MultiArray()
                msg_joints.data = data.qpos[:7].tolist()
                self.joint_pub.publish(msg_joints)
    
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