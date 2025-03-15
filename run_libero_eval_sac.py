"""
run_libero_sac_eval.py

Evaluates a trained SAC model on the LIBERO simulation environment.

Usage:
    python experiments/robot/libero/run_libero_sac_eval.py \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [libero_spatial | libero_object | libero_goal | libero_10 | libero_90] \
        --num_trials_per_task <NUM_TRIALS> \
        --use_wandb [True|False] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY> \
        --run_id_note <OPTIONAL_NOTE> \
        --seed <SEED> \
        --resize_size <IMAGE_RESIZE_SIZE>
"""

import os

import draccus

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tqdm

from stable_baselines3 import SAC

# 환경 관련 유틸리티 임포트
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    get_libero_dummy_action,
    save_rollout_video,
)
from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere

from run_libero_train_sac import LiberoGymWrapper

# LIBERO 벤치마크 로드
from LIBERO.libero.libero import benchmark


@dataclass
class EvalConfig:
    # SAC 모델 관련 파라미터
    pretrained_checkpoint: Union[str, Path] = (
        "./experiments/logs/SAC-libero_spatial-task0-2025_03_10-01_10_51_sac_model"  # 저장된 SAC 모델 체크포인트 경로
    )

    # LIBERO 환경 관련 파라미터
    task_suite_name: str = "libero_spatial"  # 태스크 스위트 (예: libero_spatial, libero_object 등)
    num_trials_per_task: int = 10  # 각 태스크당 평가 에피소드 수
    num_steps_wait: int = 10  # 평가 시작 전 시뮬레이터 안정화를 위한 대기 스텝 수
    resize_size: int = 256  # 이미지 전처리 해상도

    # 로깅 및 재현을 위한 파라미터
    seed: int = 7  # 랜덤 시드
    run_id_note: Optional[str] = None  # 추가 run id 노트 (옵션)
    local_log_dir: str = "./experiments/logs"  # 로컬 로그 저장 폴더

    # WandB 로깅 옵션
    use_wandb: bool = True  # W&B 사용 여부
    wandb_project: str = "libero-sac-v0.0.1"  # W&B 프로젝트 이름
    wandb_entity: str = "libero-spatial-sac-test-v0.0.1"  # W&B 엔터티 이름


def eval_libero(cfg: EvalConfig, model: SAC) -> None:
    # 로컬 로그 파일 초기화
    run_id = f"EVAL-SAC-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"[LOG] Logging to local file: {local_log_filepath}")

    # WandB 초기화 (옵션)
    if cfg.use_wandb:
        import wandb

        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # LIBERO task suite 초기화
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"[환경] Task suite: {cfg.task_suite_name}, Total tasks: {num_tasks}")
    log_file.write(f"Task suite: {cfg.task_suite_name}, Total tasks: {num_tasks}\n")

    total_episodes = 0
    total_successes = 0

    # 태스크별 평가 루프
    for task_id in tqdm.tqdm(range(num_tasks), desc="태스크 진행"):
        task = task_suite.get_task(task_id)
        # 각 태스크별 초기 상태 (여러 에피소드용)
        initial_states = task_suite.get_task_init_states(task_id)

        # LIBERO 환경 생성 및 래핑 (visual_rl 모드)
        env_raw, _ = get_libero_env(task, "visual_rl", resolution=cfg.resize_size)
        env = LiberoGymWrapper(env_raw, resize_size=cfg.resize_size)

        # 태스크에 따라 최대 스텝 수 설정 (기존 학습 demo 참고)
        if cfg.task_suite_name == "libero_spatial":
            max_steps = 220
        elif cfg.task_suite_name == "libero_object":
            max_steps = 280
        elif cfg.task_suite_name == "libero_goal":
            max_steps = 300
        elif cfg.task_suite_name == "libero_10":
            max_steps = 520
        elif cfg.task_suite_name == "libero_90":
            max_steps = 400
        else:
            max_steps = 220

        task_episodes = 0
        task_successes = 0

        for trial in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id} 평가"):
            total_episodes += 1
            task_episodes += 1

            # 환경 리셋 및 초기 상태 설정 (초기 상태가 있다면)
            obs, _ = env.reset()
            if len(initial_states) > trial:
                obs = env.set_init_state(initial_states[trial])

            t = 0
            replay_images = []
            done = False

            while t < max_steps + cfg.num_steps_wait:
                # 시뮬레이터 안정화를 위해 초기 num_steps_wait 스텝은 더미 액션 실행
                if t < cfg.num_steps_wait:
                    obs, _, done, _, _ = env.step(get_libero_dummy_action("sac"))
                    t += 1
                    continue

                # 이미지 전처리 및 기록 (rollout video 저장용)
                img = get_libero_image(obs, cfg.resize_size, is_openvla=False)
                replay_images.append(img)

                # 모델 입력을 위한 observation dict 구성
                observation = {"full_image": img}

                # SAC 모델을 통해 액션 예측
                action, _ = model.predict(observation)
                obs, _, done, _, _ = env.step(action)
                t += 1

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            # 평가 에피소드 종료 후 rollout 영상 저장 (옵션)
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task.name, log_file=log_file
            )

            print(f"[평가] Task {task_id}, 에피소드 {task_episodes} 성공: {done}")
            log_file.write(f"Task {task_id}, Episode {task_episodes} success: {done}\n")
            log_file.flush()

        task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0
        print(f"[평가] Task {task_id} 성공률: {task_success_rate:.3f}")
        log_file.write(f"Task {task_id} success rate: {task_success_rate:.3f}\n")
        log_file.flush()

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task.name}": task_success_rate,
                    f"num_episodes/{task.name}": task_episodes,
                }
            )

    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    print(f"[평가] 전체 성공률: {overall_success_rate:.3f}")
    log_file.write(f"Overall success rate: {overall_success_rate:.3f}\n")
    log_file.close()

    if cfg.use_wandb:
        wandb.log({"success_rate/total": overall_success_rate, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        # required=True,
        default="experiments/logs/SAC-libero_spatial-task0-2025_03_10-01_16_59_sac_model",
        help="SAC 모델 체크포인트 경로",
    )
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial", help="LIBERO task suite 이름")
    parser.add_argument("--num_trials_per_task", type=int, default=10, help="각 태스크당 평가 에피소드 수")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="시뮬레이터 안정화를 위한 대기 스텝 수")
    parser.add_argument("--use_wandb", type=bool, default=False, help="W&B 로깅 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="YOUR_WANDB_PROJECT", help="W&B 프로젝트 이름")
    parser.add_argument("--wandb_entity", type=str, default="YOUR_WANDB_ENTITY", help="W&B 엔터티 이름")
    parser.add_argument("--run_id_note", type=str, default=None, help="추가 run id 노트 (옵션)")
    parser.add_argument("--seed", type=int, default=7, help="랜덤 시드")
    parser.add_argument("--resize_size", type=int, default=256, help="이미지 전처리 해상도")
    args = parser.parse_args()

    # 재현성을 위한 랜덤 시드 설정
    set_seed_everywhere(args.seed)

    # 평가 설정 구성
    cfg = EvalConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        resize_size=args.resize_size,
        seed=args.seed,
        run_id_note=args.run_id_note,
        local_log_dir="./experiments/logs",
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # SAC 모델 로드를 위해 더미 환경 생성 (모델의 feature extractor 구성을 위해 필요)
    # 여기서는 task_suite의 첫 번째 태스크를 이용합니다.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    dummy_task = task_suite.get_task(0)
    env_raw, _ = get_libero_env(dummy_task, "visual_rl", resolution=cfg.resize_size)
    dummy_env = LiberoGymWrapper(env_raw, resize_size=cfg.resize_size)

    # 저장된 SAC 모델 로드 (env 인스턴스는 더미 환경을 사용)
    model = SAC.load(cfg.pretrained_checkpoint, env=dummy_env)
    print(f"[모델] SAC 모델이 {cfg.pretrained_checkpoint} 에서 로드되었습니다.")

    # 평가 수행
    eval_libero(cfg, model)


if __name__ == "__main__":
    main()
