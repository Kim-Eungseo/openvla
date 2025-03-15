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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 추가: LIBERO 관련 유틸리티 임포트
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.robot_utils import DATE_TIME

# LIBERO 벤치마크 (task suite) 불러오기
from LIBERO.libero.libero import benchmark


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
        img = get_libero_image(obs, self.resize_size, is_openvla=False)
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)

        # 각 키에 대해 적절한 Box 타입 설정
        self.observation_space = gym.spaces.Dict(
            {
                "full_image": gym.spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8),
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32),
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
        img = get_libero_image(obs, self.resize_size, is_openvla=False)
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)
        print("[LiberoGymWrapper] 환경 리셋 완료")

        # observation_space와 일치하는 키로 반환
        return {
            "full_image": img,
            "state": state,
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
            img = get_libero_image(obs, self.resize_size, is_openvla=False)
            state = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ).astype(np.float32)

            self._episode_ended = done
            if done:
                print("[LiberoGymWrapper] 에피소드 종료 감지 (reward={})".format(reward))

            # observation_space와 일치하는 키로 반환
            return (
                {"full_image": img, "state": state},
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
        img = get_libero_image(obs, self.resize_size, is_openvla=False)
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)
        return {"full_image": img, "state": state}

    def seed(self, seed=None):
        """
        시드를 설정합니다. Gymnasium에서는 reset 메서드에 seed를 전달하지만,
        SB3의 SubprocVecEnv와의 호환성을 위해 이 메서드를 추가합니다.

        Args:
            seed: 랜덤 시드

        Returns:
            [seed]: 설정된 시드 목록
        """
        self.env.seed(seed)
        return [seed]


# =============================================================================
# CNNProjector: Stable Baselines 3의 feature extractor (CNN + MLP)
# =============================================================================
class CNNProjector(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        """
        Args:
            observation_space: Dict형태의 observation space (키: "full_image")
            features_dim: 최종 feature vector의 차원.
        """
        super(CNNProjector, self).__init__(observation_space, features_dim)
        image_shape = observation_space.spaces["full_image"].shape  # 예: (C, H, W)

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

        # 최종 feature vector로 매핑
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations):
        # observations: dict with key "full_image"
        # 이미지 정규화: [0,255] -> [0,1]
        img = observations["full_image"].float() / 255.0
        img_features = self.cnn(img)
        return self.fc(img_features)


# =============================================================================
# WandB 로깅을 위한 콜백 클래스
# =============================================================================
class WandBLoggingCallback(BaseCallback):
    """
    WandB에 학습 과정을 로깅하기 위한 콜백
    """

    def __init__(self, verbose=0):
        super(WandBLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.step_actions = []
        self.step_rewards = []
        # 성공 횟수 추적을 위한 변수 추가
        self.total_success_count = 0

    def _on_step(self) -> bool:
        # 현재 스텝의 정보 수집
        info = self.locals.get("infos")[0] if self.locals.get("infos") else {}
        reward = self.locals.get("rewards")[0] if self.locals.get("rewards") is not None else 0
        action = self.locals.get("actions")[0] if self.locals.get("actions") is not None else None

        # 현재 에피소드 정보 업데이트
        self.current_episode_reward += reward
        self.current_episode_length += 1

        # 액션과 보상 기록
        if action is not None:
            self.step_actions.append(action)
            self.step_rewards.append(reward)

            # 액션 통계 로깅 (10 스텝마다)
            if len(self.step_actions) >= 10:
                actions = np.array(self.step_actions)
                wandb.log(
                    {
                        "actions/mean_dx": np.mean(actions[:, 0]),
                        "actions/mean_dy": np.mean(actions[:, 1]),
                        "actions/mean_dz": np.mean(actions[:, 2]),
                        "actions/mean_dRx": np.mean(actions[:, 3]),
                        "actions/mean_dRy": np.mean(actions[:, 4]),
                        "actions/mean_dRz": np.mean(actions[:, 5]),
                        "actions/mean_gripper": np.mean(actions[:, 6]),
                        "rewards/step_mean": np.mean(self.step_rewards),
                        "rewards/step_std": np.std(self.step_rewards),
                    },
                    step=self.num_timesteps,
                )
                self.step_actions = []
                self.step_rewards = []

        # 에피소드 종료 시 처리
        done = self.locals.get("dones")[0] if self.locals.get("dones") is not None else False
        if done:
            # 에피소드 정보 저장
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            success = info.get("is_success", False)
            self.episode_success.append(float(success))

            # 성공 횟수 업데이트
            if success:
                self.total_success_count += 1

            # WandB에 에피소드 정보 로깅
            wandb.log(
                {
                    "episode/reward": self.current_episode_reward,
                    "episode/length": self.current_episode_length,
                    "episode/success": float(success),
                    "episode/success_rate": np.mean(self.episode_success[-100:]) if self.episode_success else 0,
                    "episode/total_success_count": self.total_success_count,  # 누적 성공 횟수 추가
                },
                step=self.num_timesteps,
            )

            # 에피소드 정보 초기화
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # 100 에피소드마다 추가 통계 로깅
            if len(self.episode_rewards) % 10 == 0:
                wandb.log(
                    {
                        "train/mean_reward_100": np.mean(self.episode_rewards[-100:]),
                        "train/mean_length_100": np.mean(self.episode_lengths[-100:]),
                        "train/success_rate_100": np.mean(self.episode_success[-100:]),
                        "train/total_episodes": len(self.episode_rewards),
                        "train/total_success_count": self.total_success_count,  # 누적 성공 횟수 추가
                    },
                    step=self.num_timesteps,
                )

                # 모델 학습 상태 로깅
                if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                    for key, value in self.model.logger.name_to_value.items():
                        wandb.log({f"sb3/{key}": value}, step=self.num_timesteps)

        # 학습 진행 상황 로깅 (1000 스텝마다)
        if self.num_timesteps % 1000 == 0:
            # 리플레이 버퍼 통계
            if hasattr(self.model, "replay_buffer") and hasattr(self.model.replay_buffer, "pos"):
                buffer_size = self.model.replay_buffer.pos
                wandb.log(
                    {
                        "buffer/size": buffer_size,
                        "buffer/capacity_used": buffer_size / self.model.replay_buffer.buffer_size,
                    },
                    step=self.num_timesteps,
                )

            # 학습률 로깅
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                for i, param_group in enumerate(self.model.policy.optimizer.param_groups):
                    wandb.log({f"policy/learning_rate_{i}": param_group["lr"]}, step=self.num_timesteps)

            # 모델 가중치 히스토그램 (선택적)
            if self.num_timesteps % 10000 == 0:
                for name, param in self.model.policy.named_parameters():
                    if param.requires_grad:
                        wandb.log(
                            {f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())}, step=self.num_timesteps
                        )

        return True


# =============================================================================
# SAC 학습 관련 설정
# =============================================================================
@dataclass
class SACTrainConfig:
    task_suite_name: str = "libero_spatial"  # LIBERO task suite 이름
    task_id: int = 0  # 학습할 task의 ID
    total_timesteps: int = 100_000  # 총 학습 timestep 수
    seed: int = 7  # 랜덤 시드
    resize_size: int = 256  # 이미지 전처리 해상도
    buffer_size: int = 400_000  # 리플레이 버퍼 크기
    use_wandb: bool = True  # WandB 로깅 사용 여부
    wandb_project: str = "libero-sac-test-v0.0.1"  # WandB 프로젝트 이름
    wandb_entity: str = "ngseo-seoul-national-university"  # WandB 엔터티 이름
    run_id_note: str = None  # 추가 run ID 노트 (옵션)
    local_log_dir: str = "./experiments/logs"  # 로컬 모델 저장 폴더
    num_envs: int = 4  # 병렬 환경 수


# 환경 생성 함수 (멀티프로세싱용)
def make_env(task, task_suite_name, task_id, resize_size, seed, rank):
    """
    병렬 환경 생성을 위한 헬퍼 함수

    Args:
        task: LIBERO 태스크
        task_suite_name: 태스크 스위트 이름
        task_id: 태스크 ID
        resize_size: 이미지 크기
        seed: 랜덤 시드
        rank: 환경 인덱스

    Returns:
        초기화된 환경을 반환하는 함수
    """

    def _init():
        env_raw, _ = get_libero_env(task, "visual_rl", resolution=resize_size)
        env = LiberoGymWrapper(env_raw, resize_size=resize_size)
        env.seed(seed + rank)
        return env

    return _init


# =============================================================================
# 메인 함수: LIBERO 환경 구성, 래퍼 적용, SAC 학습 실행
# =============================================================================
def main():
    print("===== LIBERO SAC 학습 시작 =====")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial", help="LIBERO task suite 이름")
    parser.add_argument("--task_id", type=int, default=0, help="학습할 task의 ID")
    parser.add_argument("--total_timesteps", type=int, default=400_000, help="총 학습 timestep 수")
    parser.add_argument("--seed", type=int, default=7, help="랜덤 시드")
    parser.add_argument("--use_wandb", type=bool, default=True, help="WandB 로깅 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="libero-sac-test-v0.0.1", help="WandB 프로젝트 이름")
    parser.add_argument("--wandb_entity", type=str, default="ngseo-seoul-national-university", help="WandB 엔터티 이름")
    parser.add_argument("--run_id_note", type=str, default=None, help="추가 run ID 노트 (옵션)")
    parser.add_argument("--num_envs", type=int, default=4, help="병렬 환경 수")
    args = parser.parse_args()

    cfg = SACTrainConfig(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_id_note=args.run_id_note,
        num_envs=args.num_envs,
    )

    # MPS 디바이스 설정 (Apple Silicon GPU)
    if th.backends.mps.is_available():
        device = th.device("mps")
        print("Using MPS device")
    else:
        device = th.device("cuda")
        print("MPS is not available. Using CUDA device")

    print("[설정] task_suite_name: {}, task_id: {}".format(cfg.task_suite_name, cfg.task_id))
    print("[설정] total_timesteps: {}, seed: {}".format(cfg.total_timesteps, cfg.seed))
    print("[설정] resize_size: {}, buffer_size: {}".format(cfg.resize_size, cfg.buffer_size))
    print("[설정] num_envs: {}".format(cfg.num_envs))

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

    # 단일 환경 생성 (환경 체크용)
    print("[환경] 환경 체크를 위한 단일 환경 생성 중...")
    env_raw, _ = get_libero_env(task, "visual_rl", resolution=cfg.resize_size)
    env_check = LiberoGymWrapper(env_raw, resize_size=cfg.resize_size)

    # 환경 체크: gym 표준 인터페이스 준수 여부 확인
    print("[환경] 환경 체크 중...")
    check_env(env_check, warn=True)
    print("[환경] 환경 체크 완료")

    # 체크용 환경 닫기
    env_check.env.close()
    del env_check

    # 병렬 환경 생성
    print(f"[환경] {cfg.num_envs}개의 병렬 환경 생성 중...")
    env_fns = [
        make_env(task, cfg.task_suite_name, cfg.task_id, cfg.resize_size, cfg.seed, i) for i in range(cfg.num_envs)
    ]

    if cfg.num_envs > 1:
        env = SubprocVecEnv(env_fns)
        print(f"[환경] SubprocVecEnv로 {cfg.num_envs}개 환경 생성 완료")
    else:
        env = DummyVecEnv(env_fns)
        print("[환경] DummyVecEnv로 단일 환경 생성 완료")

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

    # WandB 로깅 콜백 생성
    if cfg.use_wandb:
        wandb_callback = WandBLoggingCallback()
        # 학습 설정 정보 로깅
        wandb.config.update(
            {
                "task_suite_name": cfg.task_suite_name,
                "task_id": cfg.task_id,
                "total_timesteps": cfg.total_timesteps,
                "seed": cfg.seed,
                "resize_size": cfg.resize_size,
                "buffer_size": cfg.buffer_size,
                "num_envs": cfg.num_envs,
                "task_name": task.name,
                "task_description": task.language_instruction if hasattr(task, "language_instruction") else "N/A",
                "model_type": "SAC",
                "device": str(device),
                "architecture": {
                    "policy": "MultiInputPolicy",
                    "features_dim": 256,
                    "cnn_layers": 3,
                },
            }
        )

        # 환경 정보 로깅
        wandb.log(
            {
                "env/num_envs": cfg.num_envs,
                "env/observation_space": str(env.observation_space),
                "env/action_space": str(env.action_space),
            }
        )

        # 모델 학습 (콜백 사용)
        model.learn(total_timesteps=cfg.total_timesteps, callback=wandb_callback)
    else:
        # 콜백 없이 학습
        model.learn(total_timesteps=cfg.total_timesteps)

    print("[학습] 학습 완료")

    # 모델 저장
    print("[저장] 모델 저장 중...")
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    model_path = os.path.join(cfg.local_log_dir, run_id + "_sac_model.zip")
    model.save(model_path)
    print("모델이 {} 에 저장되었습니다.".format(model_path))

    # 환경 닫기
    env.close()

    # 학습 완료 후 최종 성능 로깅
    if cfg.use_wandb:
        # 모델 파일 업로드
        wandb.save(model_path)

        # 최종 성능 지표 로깅
        if hasattr(wandb_callback, "episode_success") and wandb_callback.episode_success:
            final_success_rate = np.mean(wandb_callback.episode_success[-100:])
            wandb.run.summary["final_success_rate"] = final_success_rate
            wandb.run.summary["total_episodes"] = len(wandb_callback.episode_rewards)
            wandb.run.summary["total_success_count"] = wandb_callback.total_success_count
            wandb.run.summary["mean_episode_length"] = np.mean(wandb_callback.episode_lengths[-100:])
            wandb.run.summary["mean_episode_reward"] = np.mean(wandb_callback.episode_rewards[-100:])

    print("===== LIBERO SAC 학습 종료 =====")


if __name__ == "__main__":
    main()
