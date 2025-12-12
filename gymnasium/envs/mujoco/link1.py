__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import csv

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class Link1(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "link1.xml",
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        # reward_dist_weight: float = 1,   #報酬重み
        reward_control_weight: float = 1, #報酬重み
        reward_pos_weight: float = 1e+1, #報酬重み
        reward_jerk_weight: float = -5e-4, #報酬重み
        reward_vel_weight: float = -1e+1, #報酬重み
        reward_time_weight: float = -5, #報酬重み
        reward_time_designation_weight: float = -5, #報酬重み
        target_pos: float=1.57,
        eval_mode=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            # reward_dist_weight,
            reward_control_weight,
            reward_pos_weight,
            reward_jerk_weight,
            reward_vel_weight,
            reward_time_weight,
            reward_time_designation_weight,
            target_pos,
            **kwargs,
        )

        # self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        self._reward_pos_weight = reward_pos_weight
        self._reward_jerk_weight = reward_jerk_weight
        self._reward_vel_weight = reward_vel_weight
        self._reward_time_weight = reward_time_weight
        self._reward_time_designation_weight=reward_time_designation_weight
        self._target_pos=target_pos
        self.eval_mode=eval_mode
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        if self.eval_mode==True: 
            with open('/mujoco/myenv/runs/torch/eval/joint.csv','a', encoding= 'utf-8') as f:
                writer = csv.writer(f)
                # ヘッダー行を書き込む
                writer.writerow([self.data.qpos[0],self.data.qvel[0]])
        self.nowtime+=1
        preacc=self.data.qacc.copy() #1タイムステップ前の加速度(躍度の計算で使用)
        self.do_simulation(action, self.frame_skip) #行動によって状態更新
        observation = self._get_obs()
        termination=bool(self.state_vector()[0]>self._target_pos or self.nowtime==1000)  #終了条件

        # termination=bool(self.state_vector()[0]>self._target_pos)  #終了条件
        reward, reward_info = self._get_rew(action,preacc,termination)  #actionはxmlファイルのactuatorで定義された関節トルク
        info = reward_info
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        # print(type())
        # with open('/home2/isaac_env/h.csv', 'a' , encoding= 'utf-8' ) as f:  # モデルを評価する際、関節角度を格納
        #     print(h[0],file=f)

        return observation, reward, termination, False, info

    def _get_rew(self, action,preacc,termination):
        self.jerk=abs(self.data.qacc-preacc)/self.dt
        self.sum_jerk+=self.jerk
        reward_sum_jerk=(self.sum_jerk[0]*int(termination)*self._reward_jerk_weight)/self.nowtime #総躍度
        reward_pos =(self.data.qpos[0]-self._target_pos)*self._reward_pos_weight #手先をターゲットに移動する
        reward_vel=self.data.qvel[0]*int(termination)*self._reward_vel_weight
        reward_time=self.nowtime*int(termination)*self._reward_time_weight
        reward_time_designation=abs(self.nowtime-60)*int(termination)*self._reward_time_designation_weight
        # print(self.nowtime)
        # reward=reward_pos+reward_sum_jerk+reward_vel+reward_time+int(termination)*1000
        reward=reward_pos+reward_sum_jerk+reward_vel+reward_time_designation+int(termination)*300

        # 学習時に報酬を出力
        if self.eval_mode==False: 
            with open('/mujoco/myenv/runs/torch/testd/reward.csv','a', encoding= 'utf-8') as f:
                writer = csv.writer(f)
                # ヘッダー行を書き込む
                writer.writerow([reward_pos,reward_sum_jerk,reward_vel,reward_time_designation])
        reward_info = {
            # "reward_ctrl": reward_ctrl,
            "reward_pos": reward_pos,
            "reward_sum_jerk": reward_sum_jerk,
            "reward_vel": reward_vel,
            # "reward_time":reward_time,
            "reward_time_designation":reward_time_designation,
        }
        # print(self.state_vector()[1])# jointの位置、速度を表示
        return reward, reward_info

    def reset_model(self):
        qpos =self.init_qpos
        qvel = self.init_qvel 
        self.nowtime=0
        # if self.eval_mode==True: 
        #     qpos =self.init_qpos
        #     qvel = self.init_qvel 
        # else:
        #     qpos = (
        #         self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        #         + self.init_qpos
        #     )
        #     qvel = self.init_qvel + self.np_random.uniform(
        #         low=-0.005, high=0.005, size=self.model.nv
        #     )
        #     qvel[-2:] = 0
            # print("b",qpos)
        self.set_state(qpos, qvel)
        self.sum_jerk=0
        # print(qpos)
        return self._get_obs()

    # 改良版(状態)
    def _get_obs(self):
            theta = self.data.qpos.flatten()
            return np.concatenate(
                [
                    np.cos(theta),
                    np.sin(theta),
                    self.data.qpos.flatten(),
                    self.data.qvel.flatten(),
                ]
            )
    

