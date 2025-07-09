__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class HitEnv(MujocoEnv, utils.EzPickle):
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
        # xml_file: str = "reacher.xml",
        xml_file: str = "hit_env.xml",
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,   #報酬重み
        reward_control_weight: float = 1, #報酬重み
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

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
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)  #actionはxmlファイルのactuatorで定義された関節トルク
        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, action):
        # vec = self.get_body_com("fingertip") - self.get_body_com("target")
        # reward_dist = -np.linalg.norm(vec) * self._reward_dist_weight # ターゲットと手先の距離
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight # 行動をなるべく小さくする

        # reward = reward_dist + reward_ctrl

        # reward_info = {
        #     "reward_dist": reward_dist,
        #     "reward_ctrl": reward_ctrl,
        # }
        # print(self.dt)
        reward=reward_ctrl
        reward_info = {
            "reward_ctrl": reward_ctrl,
        }
        # print(self.state_vector())# jointの位置、速度を表示
        return reward, reward_info

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    # 改良版
    def _get_obs(self):
            theta = self.data.qpos.flatten()
            return np.concatenate(
                [
                    np.cos(theta),
                    np.sin(theta),
                    self.data.qpos.flatten(),
                    self.data.qvel.flatten(),
                    self.data.qacc.flatten()
                ]
            )