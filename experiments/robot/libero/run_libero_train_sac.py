"""
run_libero_sac.py

Trains an RL agent on the LIBERO simulation environment using Stable Baselines 3's SAC.
A custom CNN projector is used as a feature extractor for processing image observations.

Usage:
    python experiments/robot/libero/run_libero_sac.py \
        --task_suite_name <libero_spatial | libero_object | libero_goal | libero_10 | libero_90> \
        --task_id <TASK_ID> \
        --total_timesteps <TOTAL_TIMESTEPS> \
        --seed <SEED> \
        --use_wandb <True|False> \
        --wandb_project <WANDB_PROJECT> \
        --wandb_entity <WANDB_ENTITY> \
        --run_id_note <OPTIONAL_NOTE>
"""

import os
import sys

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

import gymnasium as gym
import argparse
import random
import numpy as np
import torch as th
import torch.nn as nn
import wandb

from dataclasses import dataclass
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

# 추가: LIBERO 관련 유틸리티 임포트
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.robot_utils import DATE_TIME

# LIBERO 벤치마크 (task suite) 불러오기
from libero.libero import benchmark


# =============================================================================
# LiberoGymWrapper: LIBERO 환경의 관측(observation)을 dict로 변환
# =============================================================================
class LiberoGymWrapper(gym.Env):
    """Gymnasium 환경 래퍼 클래스"""

    metadata = {"render_modes": ["human"]}

    # 스텝 카운터 추가
    step_counter = 0

    def __init__(self, env, resize_size):
        """
        Args:
            env: 원본 LIBERO 환경.
            resize_size: 이미지 전처리에 사용할 해상도.
        """
        super().__init__()
        self.env = env
        self.resize_size = resize_size
        self._episode_ended = False

        # 초기 관측을 통해 observation space를 정의합니다.
        obs = self.env.reset()
        img = get_libero_image(obs, self.resize_size)
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)

        # 각 키에 대해 적절한 Box 타입 설정
        self.observation_space = gym.spaces.Dict(
            {
                "full_image": gym.spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8),
                "agentview_image": gym.spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8),
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32),
                "robot0_joint_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "robot0_joint_pos_cos": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
                "robot0_joint_pos_sin": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
                "robot0_joint_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "robot0_eef_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "robot0_eef_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "robot0_gripper_qpos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "robot0_gripper_qvel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "robot0_eye_in_hand_image": gym.spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8),
                # 추가적인 키들에 대한 Box 타입 설정
                "akita_black_bowl_1_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "akita_black_bowl_1_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "akita_black_bowl_1_to_robot0_eef_pos": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "akita_black_bowl_1_to_robot0_eef_quat": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
                "akita_black_bowl_2_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "akita_black_bowl_2_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "akita_black_bowl_2_to_robot0_eef_pos": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "akita_black_bowl_2_to_robot0_eef_quat": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
                "cookies_1_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "cookies_1_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "cookies_1_to_robot0_eef_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "cookies_1_to_robot0_eef_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "glazed_rim_porcelain_ramekin_1_pos": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "glazed_rim_porcelain_ramekin_1_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
                "plate_1_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "plate_1_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "plate_1_to_robot0_eef_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "plate_1_to_robot0_eef_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "robot0_proprio-state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
                "object-state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32),
            }
        )

        # LIBERO 환경의 action space 정의
        # 7차원 action space: [dx, dy, dz, dRx, dRy, dRz, gripper]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        print("[LiberoGymWrapper] 초기화 완료: 이미지 크기={}, 상태 차원={}".format(img.shape, state.shape))

    def reset(self, *, seed=None, options=None):
        print("[LiberoGymWrapper] 환경 리셋 시작")
        super().reset(seed=seed)
        obs = self.env.reset()
        self._episode_ended = False
        img = get_libero_image(obs, self.resize_size)
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)
        print("[LiberoGymWrapper] 환경 리셋 완료")

        # observation_space와 일치하는 키로 반환
        return {
            "agentview_image": img,
            "state": state,
            "robot0_joint_pos": obs["robot0_joint_pos"],
            "robot0_joint_pos_cos": obs["robot0_joint_pos_cos"],
            "robot0_joint_pos_sin": obs["robot0_joint_pos_sin"],
            "robot0_joint_vel": obs["robot0_joint_vel"],
            "robot0_eef_pos": obs["robot0_eef_pos"],
            "robot0_eef_quat": obs["robot0_eef_quat"],
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
            "robot0_gripper_qvel": obs["robot0_gripper_qvel"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "akita_black_bowl_1_pos": obs["akita_black_bowl_1_pos"],
            "akita_black_bowl_1_quat": obs["akita_black_bowl_1_quat"],
            "akita_black_bowl_1_to_robot0_eef_pos": obs["akita_black_bowl_1_to_robot0_eef_pos"],
            "akita_black_bowl_1_to_robot0_eef_quat": obs["akita_black_bowl_1_to_robot0_eef_quat"],
            "akita_black_bowl_2_pos": obs["akita_black_bowl_2_pos"],
            "akita_black_bowl_2_quat": obs["akita_black_bowl_2_quat"],
            "akita_black_bowl_2_to_robot0_eef_pos": obs["akita_black_bowl_2_to_robot0_eef_pos"],
            "akita_black_bowl_2_to_robot0_eef_quat": obs["akita_black_bowl_2_to_robot0_eef_quat"],
            "cookies_1_pos": obs["cookies_1_pos"],
            "cookies_1_quat": obs["cookies_1_quat"],
            "cookies_1_to_robot0_eef_pos": obs["cookies_1_to_robot0_eef_pos"],
            "cookies_1_to_robot0_eef_quat": obs["cookies_1_to_robot0_eef_quat"],
            "glazed_rim_porcelain_ramekin_1_pos": obs["glazed_rim_porcelain_ramekin_1_pos"],
            "glazed_rim_porcelain_ramekin_1_quat": obs["glazed_rim_porcelain_ramekin_1_quat"],
            "glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos": obs["glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos"],
            "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat": obs[
                "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat"
            ],
            "plate_1_pos": obs["plate_1_pos"],
            "plate_1_quat": obs["plate_1_quat"],
            "plate_1_to_robot0_eef_pos": obs["plate_1_to_robot0_eef_pos"],
            "plate_1_to_robot0_eef_quat": obs["plate_1_to_robot0_eef_quat"],
            "robot0_proprio-state": obs["robot0_proprio-state"],
            "object-state": obs["object-state"],
        }, {}

    def step(self, action):
        LiberoGymWrapper.step_counter += 1
        if LiberoGymWrapper.step_counter % 100 == 0:
            print("[LiberoGymWrapper] 스텝 {} 실행 중".format(LiberoGymWrapper.step_counter))

        if self._episode_ended:
            print("[LiberoGymWrapper] 에피소드 종료 상태에서 자동 리셋")
            obs_dict, _ = self.reset()
            return obs_dict, 0.0, False, False, {}

        try:
            obs, reward, done, info = self.env.step(action)
            img = get_libero_image(obs, self.resize_size)
            state = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ).astype(np.float32)

            self._episode_ended = done
            if done:
                print("[LiberoGymWrapper] 에피소드 종료 감지 (reward={})".format(reward))

            # observation_space와 일치하는 키로 반환
            return (
                {
                    "agentview_image": img,
                    "state": state,
                    "robot0_joint_pos": obs["robot0_joint_pos"],
                    "robot0_joint_pos_cos": obs["robot0_joint_pos_cos"],
                    "robot0_joint_pos_sin": obs["robot0_joint_pos_sin"],
                    "robot0_joint_vel": obs["robot0_joint_vel"],
                    "robot0_eef_pos": obs["robot0_eef_pos"],
                    "robot0_eef_quat": obs["robot0_eef_quat"],
                    "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                    "robot0_gripper_qvel": obs["robot0_gripper_qvel"],
                    "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
                    "akita_black_bowl_1_pos": obs["akita_black_bowl_1_pos"],
                    "akita_black_bowl_1_quat": obs["akita_black_bowl_1_quat"],
                    "akita_black_bowl_1_to_robot0_eef_pos": obs["akita_black_bowl_1_to_robot0_eef_pos"],
                    "akita_black_bowl_1_to_robot0_eef_quat": obs["akita_black_bowl_1_to_robot0_eef_quat"],
                    "akita_black_bowl_2_pos": obs["akita_black_bowl_2_pos"],
                    "akita_black_bowl_2_quat": obs["akita_black_bowl_2_quat"],
                    "akita_black_bowl_2_to_robot0_eef_pos": obs["akita_black_bowl_2_to_robot0_eef_pos"],
                    "akita_black_bowl_2_to_robot0_eef_quat": obs["akita_black_bowl_2_to_robot0_eef_quat"],
                    "cookies_1_pos": obs["cookies_1_pos"],
                    "cookies_1_quat": obs["cookies_1_quat"],
                    "cookies_1_to_robot0_eef_pos": obs["cookies_1_to_robot0_eef_pos"],
                    "cookies_1_to_robot0_eef_quat": obs["cookies_1_to_robot0_eef_quat"],
                    "glazed_rim_porcelain_ramekin_1_pos": obs["glazed_rim_porcelain_ramekin_1_pos"],
                    "glazed_rim_porcelain_ramekin_1_quat": obs["glazed_rim_porcelain_ramekin_1_quat"],
                    "glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos": obs[
                        "glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos"
                    ],
                    "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat": obs[
                        "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat"
                    ],
                    "plate_1_pos": obs["plate_1_pos"],
                    "plate_1_quat": obs["plate_1_quat"],
                    "plate_1_to_robot0_eef_pos": obs["plate_1_to_robot0_eef_pos"],
                    "plate_1_to_robot0_eef_quat": obs["plate_1_to_robot0_eef_quat"],
                    "robot0_proprio-state": obs["robot0_proprio-state"],
                    "object-state": obs["object-state"],
                },
                reward,
                done,
                False,
                info,
            )

        except ValueError as e:
            print("[LiberoGymWrapper] 예외 발생: {}".format(str(e)))
            if "executing action in terminated episode" in str(e):
                print("[LiberoGymWrapper] 종료된 에피소드에서 액션 실행 시도, 자동 리셋")
                self._episode_ended = True
                obs_dict, _ = self.reset()
                return obs_dict, 0.0, False, False, {}
            else:
                raise e

    def render(self):
        return self.env.render()

    def set_init_state(self, init_state):
        """
        초기 상태를 설정합니다.

        Args:
            init_state: 초기 상태 정보

        Returns:
            관측값 (observation)
        """
        print("[LiberoGymWrapper] 초기 상태 설정")
        # 원본 환경의 set_init_state 메서드 호출
        obs = self.env.set_init_state(init_state)
        self._episode_ended = False

        # 관측값을 dict 형태로 변환
        img = get_libero_image(obs, self.resize_size)
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)
        return {
            "agentview_image": img,
            "state": state,
            "robot0_joint_pos": obs["robot0_joint_pos"],
            "robot0_joint_pos_cos": obs["robot0_joint_pos_cos"],
            "robot0_joint_pos_sin": obs["robot0_joint_pos_sin"],
            "robot0_joint_vel": obs["robot0_joint_vel"],
            "robot0_eef_pos": obs["robot0_eef_pos"],
            "robot0_eef_quat": obs["robot0_eef_quat"],
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
            "robot0_gripper_qvel": obs["robot0_gripper_qvel"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "akita_black_bowl_1_pos": obs["akita_black_bowl_1_pos"],
            "akita_black_bowl_1_quat": obs["akita_black_bowl_1_quat"],
            "akita_black_bowl_1_to_robot0_eef_pos": obs["akita_black_bowl_1_to_robot0_eef_pos"],
            "akita_black_bowl_1_to_robot0_eef_quat": obs["akita_black_bowl_1_to_robot0_eef_quat"],
            "akita_black_bowl_2_pos": obs["akita_black_bowl_2_pos"],
            "akita_black_bowl_2_quat": obs["akita_black_bowl_2_quat"],
            "akita_black_bowl_2_to_robot0_eef_pos": obs["akita_black_bowl_2_to_robot0_eef_pos"],
            "akita_black_bowl_2_to_robot0_eef_quat": obs["akita_black_bowl_2_to_robot0_eef_quat"],
            "cookies_1_pos": obs["cookies_1_pos"],
            "cookies_1_quat": obs["cookies_1_quat"],
            "cookies_1_to_robot0_eef_pos": obs["cookies_1_to_robot0_eef_pos"],
            "cookies_1_to_robot0_eef_quat": obs["cookies_1_to_robot0_eef_quat"],
            "glazed_rim_porcelain_ramekin_1_pos": obs["glazed_rim_porcelain_ramekin_1_pos"],
            "glazed_rim_porcelain_ramekin_1_quat": obs["glazed_rim_porcelain_ramekin_1_quat"],
            "glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos": obs["glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos"],
            "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat": obs[
                "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat"
            ],
            "plate_1_pos": obs["plate_1_pos"],
            "plate_1_quat": obs["plate_1_quat"],
            "plate_1_to_robot0_eef_pos": obs["plate_1_to_robot0_eef_pos"],
            "plate_1_to_robot0_eef_quat": obs["plate_1_to_robot0_eef_quat"],
            "robot0_proprio-state": obs["robot0_proprio-state"],
            "object-state": obs["object-state"],
        }


# =============================================================================
# CNNProjector: Stable Baselines 3의 feature extractor (CNN + MLP)
# =============================================================================
class CNNProjector(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        """
        Args:
            observation_space: Dict형태의 observation space (키: "full_image", "state")
            features_dim: 최종 feature vector의 차원.
        """
        super(CNNProjector, self).__init__(observation_space, features_dim)
        image_shape = observation_space.spaces["full_image"].shape  # 예: (C, H, W)
        state_shape = observation_space.spaces["state"].shape  # 예: (d,)

        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 임의의 입력을 통해 CNN 출력 차원 계산
        with th.no_grad():
            sample_img = th.as_tensor(np.zeros((1, *image_shape), dtype=np.float32))
            n_flatten = self.cnn(sample_img).shape[1]

        # MLP for state processing
        self.state_mlp = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.ReLU(),
        )
        # CNN과 MLP의 출력을 결합한 후, 최종 feature vector로 매핑
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations):
        # observations: dict with keys "full_image" and "state"
        # 이미지 정규화: [0,255] -> [0,1]
        img = observations["full_image"].float() / 255.0
        img_features = self.cnn(img)
        state = observations["state"].float()
        state_features = self.state_mlp(state)
        combined = th.cat([img_features, state_features], dim=1)
        return self.fc(combined)


# =============================================================================
# SAC 학습 관련 설정
# =============================================================================
@dataclass
class SACTrainConfig:
    task_suite_name: str = "libero_spatial"  # LIBERO task suite 이름
    task_id: int = 0  # 학습할 task의 ID
    total_timesteps: int = 1000  # 총 학습 timestep 수
    seed: int = 7  # 랜덤 시드
    resize_size: int = 128  # 이미지 전처리 해상도 (256에서 128로 축소)
    buffer_size: int = 100000  # 리플레이 버퍼 크기
    use_wandb: bool = False  # WandB 로깅 사용 여부
    wandb_project: str = "YOUR_WANDB_PROJECT"  # WandB 프로젝트 이름
    wandb_entity: str = "YOUR_WANDB_ENTITY"  # WandB 엔터티 이름
    run_id_note: str = None  # 추가 run ID 노트 (옵션)
    local_log_dir: str = "./experiments/logs"  # 로컬 모델 저장 폴더


# =============================================================================
# 메인 함수: LIBERO 환경 구성, 래퍼 적용, SAC 학습 실행
# =============================================================================
def main():
    print("===== LIBERO SAC 학습 시작 =====")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial", help="LIBERO task suite 이름")
    parser.add_argument("--task_id", type=int, default=0, help="학습할 task의 ID")
    parser.add_argument("--total_timesteps", type=int, default=1000, help="총 학습 timestep 수")
    parser.add_argument("--seed", type=int, default=7, help="랜덤 시드")
    parser.add_argument("--use_wandb", type=bool, default=False, help="WandB 로깅 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="YOUR_WANDB_PROJECT", help="WandB 프로젝트 이름")
    parser.add_argument("--wandb_entity", type=str, default="YOUR_WANDB_ENTITY", help="WandB 엔터티 이름")
    parser.add_argument("--run_id_note", type=str, default=None, help="추가 run ID 노트 (옵션)")
    args = parser.parse_args()

    cfg = SACTrainConfig(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        resize_size=128,  # 이미지 크기 축소
        buffer_size=100000,  # 리플레이 버퍼 크기 제한
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_id_note=args.run_id_note,
    )

    # MPS 디바이스 설정 (Apple Silicon GPU)
    if th.backends.mps.is_available():
        device = th.device("mps")
        print("Using MPS device")
    else:
        device = th.device("cpu")
        print("MPS is not available. Using CPU")

    print("[설정] task_suite_name: {}, task_id: {}".format(cfg.task_suite_name, cfg.task_id))
    print("[설정] total_timesteps: {}, seed: {}".format(cfg.total_timesteps, cfg.seed))
    print("[설정] resize_size: {}, buffer_size: {}".format(cfg.resize_size, cfg.buffer_size))

    # 랜덤 시드 설정
    set_random_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    print("[설정] 랜덤 시드 설정 완료: {}".format(cfg.seed))

    # run_id 구성
    run_id = f"SAC-{cfg.task_suite_name}-task{cfg.task_id}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    print("[설정] run_id: {}".format(run_id))
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_id)
        print("[설정] WandB 초기화 완료")

    # LIBERO task suite에서 task 선택
    print("[환경] LIBERO 벤치마크 로드 중...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    print("[환경] 태스크 로드 완료: {}".format(task.name))

    # LIBERO 환경 생성 (visual_rl 용으로 지정)
    print("[환경] LIBERO 환경 생성 중...")
    env_raw, _ = get_libero_env(task, "visual_rl", resolution=cfg.resize_size)
    env = LiberoGymWrapper(env_raw, resize_size=cfg.resize_size)

    # (옵션) 환경 체크: gym 표준 인터페이스 준수 여부 확인
    print("[환경] 환경 체크 중...")
    check_env(env, warn=True)
    print("[환경] 환경 체크 완료")

    # SAC의 policy_kwargs에 커스텀 feature extractor (CNNProjector) 등록
    print("[모델] SAC 모델 설정 중...")
    policy_kwargs = dict(
        features_extractor_class=CNNProjector,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # MultiInputPolicy를 사용해 SAC 모델 생성 (관측이 dict 형태이므로)
    print("[모델] SAC 모델 생성 중...")
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg.seed,
        buffer_size=cfg.buffer_size,  # 리플레이 버퍼 크기 제한
        device=device,  # MPS 디바이스 사용
    )
    print("[모델] SAC 모델 생성 완료")

    # 학습 시작
    print("[학습] 학습 시작 (total_timesteps: {})...".format(cfg.total_timesteps))
    model.learn(total_timesteps=cfg.total_timesteps)
    print("[학습] 학습 완료")

    # 모델 저장
    print("[저장] 모델 저장 중...")
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    model_path = os.path.join(cfg.local_log_dir, run_id + "_sac_model.zip")
    model.save(model_path)
    print("모델이 {} 에 저장되었습니다.".format(model_path))
    print("===== LIBERO SAC 학습 종료 =====")


if __name__ == "__main__":
    main()
