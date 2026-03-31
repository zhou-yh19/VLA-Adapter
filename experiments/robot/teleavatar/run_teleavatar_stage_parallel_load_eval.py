"""
run_teleavatar_eval.py

Evaluates a trained policy on a Teleavatar.

```bash
python experiments/robot/teleavatar/run_teleavatar_parallel_load_eval.py   --pretrained_checkpoint outputs/Teleavatar-stuffed-animal-1   > eval_logs/shihaoran--teleavatar--stuffed_animal-1--chkpt.log 2>&1 &
```
"""

import json
import logging
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
import queue  # 用于线程安全队列
import threading  # 用于多线程

import draccus
import numpy as np
import tqdm
import time
import random

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_stage_classifier,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    # get_action,
    get_stage_action,
    get_model,
    set_seed_everywhere,
)
from prismatic.vla.constants import TELEAVATAR_CONSTANTS
PROPRIO_DIM = TELEAVATAR_CONSTANTS["PROPRIO_DIM"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Import Robot interface
from experiments.robot.teleavatar.robot_interface import TeleavatarRobotInterface



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_action_horizon: int = 30                     # Number of actions in each chunk returned by policy
    num_open_loop_steps: int = 25                    # Number of actions to execute before querying policy again
    action_generation_frequency: int = 10            # Action generation frequency in Hz
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Teleavatar runtime parameters
    #################################################################################################################
    control_frequency: float = 30                    # Control loop frequency in Hz               
    task_description: str = "Organize_the_Desk." 
                                                     # Language instruction for the robot
    num_episodes: int = 1                           # Number of episodes to run
    max_episode_steps: int = 1800                     # Maximum VLA inference count per episode 
                                                     # action_generation_frequency * one_episode_duration(30s)
                                                     # max_episode_steps = 600时，相当于一个episode会运行1分钟

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)
    use_parallel_loading: bool = False                # If True, load components in parallel to speed up initialization
    llm_dim: int = 896                               # LLM dimension
    act_m: float = 0.1                               # Action weight decay coefficient
    enable_stage_manager: bool = False                # If False, disable stage manager; low_level_task stays all-Waiting
    mean_voting_flag: bool = True                   # If True, use mean voting to update stage_progress, else use weighted voting
    stage_insert_between_tasks: bool = True          # If True, insert stage_queries between HL and LL prompts; else before action tokens
                                                    # 只有 vla-adapter-stage4 之后的模型是这个架构

    # fmt: on
    save_version: str = "vla-adapter-teleavatar-stage"     # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"



def initialize_model(cfg: GenerateConfig, log_file):
    """
    Initialize model and associated components with full parallel loading.
    
    Args:
        cfg: Configuration object
        log_file: Optional log file for logging
    
    Returns:
        Tuple of (model, action_head, proprio_projector, noisy_action_projector, processor, stage_classifier)
    """
    noisy_action_projector = None
    
    if cfg.use_parallel_loading:
        # 完全并行加载：主模型、proprio_projector、action_head、processor、stage_classifier 同时加载
        # 使用 ThreadPoolExecutor 实现并行加载
        # 并行加载时必须关闭 low_cpu_mem_usage，否则主模型会使用 meta device 创建张量，
        # 导致其他线程里创建的 action_head 等模块也变成 meta 张量，进而 .to(device) 报错
        cfg.low_cpu_mem_usage = False
        max_workers = 1  # llm
        futures = {}
        
        # 计算需要的线程数
        if cfg.model_family == "openvla":
            max_workers += 1  # processor
        if cfg.use_proprio:
            max_workers += 1  # proprio_projector
        if cfg.use_l1_regression:
            max_workers += 1  # action_head
        max_workers += 1  # stage_classifier (vla-adapter-stage，若 checkpoint 中存在则加载)
        
        log_message(f"开始并行加载，使用 {max_workers} 个线程", log_file)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交主模型加载任务
            futures[executor.submit(get_model, cfg)] = 'model'
            log_message("已提交 llm 加载任务", log_file)
            
            # 提交 processor 加载任务（如果使用 openvla）
            processor = None
            if cfg.model_family == "openvla":
                futures[executor.submit(get_processor, cfg)] = 'processor'
                log_message("已提交 processor 加载任务", log_file)
            
            # 提交 proprio_projector 加载任务（如果启用）
            proprio_projector = None
            if cfg.use_proprio:
                futures[executor.submit(
                    get_proprio_projector,
                    cfg,
                    cfg.llm_dim,
                    PROPRIO_DIM,  # 14-dimensional proprio for Teleavatar
                )] = 'proprio'
                log_message("已提交 proprio_projector 加载任务", log_file)
            
            # 提交 action_head 加载任务（如果启用）
            action_head = None
            if cfg.use_l1_regression:
                futures[executor.submit(
                    get_action_head,
                    cfg,
                    cfg.llm_dim,
                )] = 'action_head'
                log_message("已提交 action_head 加载任务", log_file)
            
            # 提交 stage_classifier 加载任务（vla-adapter-stage，从 run_* 目录加载）
            stage_classifier = None
            futures[executor.submit(get_stage_classifier, cfg, cfg.llm_dim)] = 'stage_classifier'
            log_message("已提交 stage_classifier 加载任务", log_file)
            
            # 等待所有任务完成并收集结果
            completed_count = 0
            model = None  # 初始化 model 变量
            
            for future in as_completed(futures):
                component_type = futures[future]
                try:
                    result = future.result()
                    completed_count += 1
                    
                    if   component_type == 'model':
                        model = result
                        # 主模型加载完成后，设置版本和检查 unnorm_key
                        model.set_version(cfg.save_version)
                        if cfg.model_family == "openvla":
                            check_unnorm_key(cfg, model)
                        log_message(f"主模型加载完成 ({completed_count}/{len(futures)})", log_file)
                    elif component_type == 'processor':
                        processor = result
                        log_message(f"processor 加载完成 ({completed_count}/{len(futures)})", log_file)
                    elif component_type == 'proprio':
                        proprio_projector = result
                        log_message(f"proprio_projector 加载完成 ({completed_count}/{len(futures)})", log_file)
                    elif component_type == 'action_head':
                        action_head = result
                        log_message(f"action_head 加载完成 ({completed_count}/{len(futures)})", log_file)
                    elif component_type == 'stage_classifier':
                        stage_classifier = result
                        if stage_classifier is not None:
                            log_message(f"stage_classifier (vla-adapter-stage) 加载完成 ({completed_count}/{len(futures)})", log_file)
                        else:
                            log_message(f"未找到 stage_classifier 权重，跳过 ({completed_count}/{len(futures)})", log_file)
                except Exception as e:
                    log_message(f"加载 {component_type} 时出错: {e}", log_file)
                    raise
            
            # 验证主模型已成功加载
            if model is None:
                raise RuntimeError("主模型加载失败！")
            
            log_message(f"所有组件加载完成！共 {len(futures)} 个组件", log_file)
        
        return model, action_head, proprio_projector, noisy_action_projector, processor, stage_classifier
    else:
        # 串行加载（原始方式）
        # Load model
        model = get_model(cfg)
        model.set_version(cfg.save_version)

        # Get OpenVLA processor if needed
        processor = None
        if cfg.model_family == "openvla":
            processor = get_processor(cfg)
            check_unnorm_key(cfg, model)
        
        # Load proprio projector if needed
        proprio_projector = None
        if cfg.use_proprio:
            proprio_projector = get_proprio_projector(
                cfg,
                cfg.llm_dim,
                proprio_dim=PROPRIO_DIM,
            )

        # Load action head if needed
        action_head = None
        if cfg.use_l1_regression:
            action_head = get_action_head(cfg, cfg.llm_dim)

        # Load stage classifier (vla-adapter-stage) if checkpoint contains it
        stage_classifier = get_stage_classifier(cfg, cfg.llm_dim)
        if stage_classifier is not None:
            log_message("stage_classifier (vla-adapter-stage) 加载完成", log_file)
        else:
            log_message("未找到 stage_classifier 权重，跳过", log_file)

        return model, action_head, proprio_projector, noisy_action_projector, processor, stage_classifier


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Set the unnorm_key in cfg
    cfg.unnorm_key = list(model.norm_stats.keys())[0]



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-Teleavatar-Grab_stuffed_animal-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()



def _get_teleavatar_chest_image(obs):
    """Get chest raw-image from ros2 interface."""
    return obs['images']['head_camera']

def _get_teleavatar_left_wrist_image(obs):
    """Get left-wrist raw-image from ros2 interface."""
    return obs['images']['left_color']

def _get_teleavatar_right_wrist_image(obs):
    """Get right-wrist raw-image from ros2 interface."""
    return obs['images']['right_color']

def _get_teleavatar_state(obs):
    """Get proprio from ros2 interface and normalize"""
    return obs['state']
    
def prepare_observation(obs):
    """Prepare observation for policy input."""
    # Get raw images
    chest_img = _get_teleavatar_chest_image(obs)
    left_wrist_img = _get_teleavatar_left_wrist_image(obs)
    right_wrist_img = _get_teleavatar_right_wrist_image(obs)

    # Get state
    state = _get_teleavatar_state(obs)

    # Prepare observations dict
    observation = {
        "full_image": chest_img,
        "left_wrist_image": left_wrist_img,
        "right_wrist_image": right_wrist_img,
        "state": state,
    }

    return observation  # Return both processed observation and original image for replay



def prepare_task_description(cfg: GenerateConfig) -> str:
    """Prepare task description for policy input."""
    ts = cfg.task_description.replace("_", " ")
    return ts + " and Do not move the yellow cup. " + \
    "Task1: placing the cleaning cloth in the right-side box. " + \
    "Task2: putting the marker into the center pen holder. "+\
    "Task3: placing the block on the second shelf of the left cabinet. "+\
    "Task4: placing the candle on the second shelf of the left cabinet."



def update_stage_progress(
    stage_progress: dict,
    stage_id_arr: np.ndarray,
    weights: np.ndarray,
    mean_voting_flag: bool = False,
) -> dict:
    """
    利用投票机制来更新 stage_progress
    每个 Task 都会经历三个状态："Waiting", "Active", "Done"
    同时有且只有一个 Task 处于 Active 状态
    """
    # 加权投票
    if mean_voting_flag is not True:
        weighted_score = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for idx, stage_id in enumerate(stage_id_arr):
            weighted_score[stage_id] += weights[idx]

        # “有效阶段”为加权得分最高的 stage_id
        effective_stage = max(weighted_score, key=weighted_score.get)
        if stage_progress[effective_stage] == "Waiting":
            # 当前任务需要开始处理，关闭其他正在处理的任务
            for stage_id, stage_state in stage_progress.items():
                if stage_state == "Active":
                    stage_progress[stage_id] = "Done"
                    break
            stage_progress[effective_stage] = "Active"
        elif stage_progress[effective_stage] == "Active":
            # 当前任务正在处理中，无需更新
            pass
        elif stage_progress[effective_stage] == "Done":
            # 当前阶段已在之前完成，无需更新
            pass
        return stage_progress
    
    # 均值投票
    else:
        weighted_score = {1: 0, 2: 0, 3: 0, 4: 0}
        for stage_id in stage_id_arr:
            weighted_score[stage_id] += 1

        # 找到所有得分等于最高分的 stage_id
        max_score = max(weighted_score.values())
        best_stage_ids = [stage_id for stage_id, score in weighted_score.items() if score == max_score]

        # 先删除其中是Done的stage_id
        best_stage_ids = list(filter(lambda stage_id: stage_progress[stage_id] != "Done", best_stage_ids))
        if best_stage_ids == []:
            return stage_progress

        # 然后判断其中是否有Active，如果有就不需要改变
        for stage_id in best_stage_ids:
            if stage_progress[stage_id] == "Active":
                return stage_progress

        # 如果没有Active，此时其中的stage_id都是waiting，随机选择一个作为Active
        effective_stage = random.choice(best_stage_ids)
        # 当前任务需要开始处理，关闭其他正在处理的任务
        for stage_id, stage_state in stage_progress.items():
            if stage_state == "Active":
                stage_progress[stage_id] = "Done"
                break
        stage_progress[effective_stage] = "Active"
        return stage_progress



def run_episode(
    cfg: GenerateConfig,
    robot_interface: TeleavatarRobotInterface,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
    inference_time_ls: list[float] = [],
    stage_classifier=None,
):
    """Run a single episode in the environment with parallel action generation and publishing."""
    # 里面存放多个action_queue，但只有前九个甚至更少能够参与到当前时刻的动作生成中
    action_stage_queues = list()
    # 记录一个episode中四个stage的进度
    stage_progress = {1:"Waiting", 2:"Waiting", 3:"Waiting", 4:"Waiting"}
    
    # 控制线程的停止标志
    stop_event = threading.Event()
    exception_occurred = threading.Event()
    exception_info = [None]  # 用于存储异常信息
    
    # 创建同步屏障，确保两个线程同时开始（需要2个线程）
    start_barrier = threading.Barrier(2)
    # 创建停止屏障，确保两个线程同时停止（需要2个线程）
    stop_barrier = threading.Barrier(2)
    
    # 动作生成频率：10Hz = 0.1秒间隔
    action_generation_interval = 1.0 / cfg.action_generation_frequency  # 使用配置中的频率
    # 动作发布频率：30Hz = 0.033秒间隔
    action_publish_interval = 1.0 / cfg.control_frequency  # 30Hz
    
    inference_count = [0]  # 使用列表以便在线程间共享
    publish_count = [0]    # 发布步数；日志前缀 [GEN]/[STAGE]/[PUB] 便于 grep 分别查看队列与阶段

    # stage_progress is shared between generation & publishing threads (stage_lock)
    stage_log_buffer = []
    stage_log_last_flush = [time.time()]
    stage_lock = threading.Lock()

    def action_generation_thread():
        """以10Hz频率生成动作并放入队列的线程"""
        try:
            start_barrier.wait() # 等待两个线程都准备好后同时开始

            # 生成线程只因为 stop_event 被设置而停止，由发布线程控制停止时机
            # 安全检查：如果达到最大步数，记录警告但继续运行（由发布线程决定何时停止）
            print_generation_stop_flag = False

            # 将子任务内容写入到提示词中
            task_description = prepare_task_description(cfg)

            while not stop_event.is_set():
                # 安全检查：如果达到最大步数，等待发布线程发出停止信号
                if inference_count[0] >= cfg.max_episode_steps:
                    if print_generation_stop_flag is False:
                        log_message(f"警告: 已达到最大步数 {cfg.max_episode_steps}，但继续运行等待发布线程停止信号", log_file)
                        print_generation_stop_flag = True
                    time.sleep(action_publish_interval)
                    continue

                # 正常执行生成动作程序
                generation_start_time = time.time()

                # 将stage_progress转换为low_level_task
                if cfg.enable_stage_manager:
                    with stage_lock:
                        low_level_task = ",".join(
                            f"Task{sid}:{stage_progress[sid]}" for sid in range(1, len(stage_progress) + 1)
                        )
                else:
                    low_level_task = "Task1:Waiting,Task2:Waiting,Task3:Waiting,Task4:Waiting"
                
                # 获取观察
                obs = robot_interface.get_observation()
                observation = prepare_observation(obs)

                # 生成动作（若加载了 stage_classifier 则同时返回当前 stage id 0/1/2/3）
                inference_start_time = time.time()
                actions, stage_probs, num_patches, num_prompt_tokens = get_stage_action(
                    cfg,
                    model=model,
                    obs=observation,
                    high_level_task=task_description,
                    low_level_task=low_level_task,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                    stage_classifier=stage_classifier,
                )
                inference_end_time = time.time()
                inference_time_ls.append(inference_end_time - inference_start_time)
                
                # 记录图片patch数量和prompt token数量
                log_message(
                    f"num_patches: {num_patches}, num_prompt_tokens: {num_prompt_tokens}", 
                    log_file,
                )
                
                # 将 (action, stage_id) 序列放入动作队列中
                max_index, _ = max(enumerate(stage_probs), key=lambda x: x[1])
                stage_id = max_index + 1  # 将 idx 转换为 stage_id 
                action_stage_pairs = [(action, stage_id) for action in actions]
                action_stage_queues.append(action_stage_pairs)
                inference_count[0] += 1

                # Accumulate stage snapshots; flush to log every 10 seconds
                queue_lens = [len(q) for q in action_stage_queues]
                with stage_lock:
                    stage_snapshot = dict(stage_progress)
                    stage_log_buffer.append(
                        f"xstep={inference_count[0]} | stage_progress={stage_snapshot} | queues={len(action_stage_queues)} {queue_lens}"
                    )
                    now = time.time()
                    if now - stage_log_last_flush[0] >= 10.0:
                        log_message(
                            f"[stage_log 10s summary] ({len(stage_log_buffer)} entries):\n"
                            + "\n".join(stage_log_buffer),
                            log_file,
                        )
                        stage_log_buffer.clear()
                        stage_log_last_flush[0] = now
                
                # 控制生成频率：10Hz
                generation_end_time = time.time()
                elapsed_interval = generation_end_time - generation_start_time
                if action_generation_interval > elapsed_interval:
                    time.sleep(action_generation_interval - elapsed_interval)
            
            # Flush remaining stage log entries before exiting
            with stage_lock:
                if stage_log_buffer:
                    log_message(
                        f"[stage_log final flush] ({len(stage_log_buffer)} entries):\n"
                        + "\n".join(stage_log_buffer),
                        log_file,
                    )
                    stage_log_buffer.clear()

            # 生成线程检测到 stop_event 后（由发布线程设置），等待同步停止
            log_message("生成线程检测到停止标志，准备停止...", log_file)
            stop_barrier.wait()  # 等待两个线程都准备好后同时停止
                    
        except Exception as e:
            log_message(f"动作生成线程错误: {e}", log_file)
            exception_info[0] = e
            exception_occurred.set()
            stop_event.set()
            try:
                stop_barrier.wait(timeout=0.1)  # 尝试同步停止，但不要阻塞太久
            except threading.BrokenBarrierError:
                pass  # 如果另一个线程已经停止，忽略错误
    
    def action_publishing_thread():
        """以 30Hz 频率从队列取出动作并发布的线程"""
        try:
            start_barrier.wait() # 等待两个线程都准备好后同时开始

            # 等待模型发出第一个动作
            while len(action_stage_queues) == 0:
                time.sleep(action_publish_interval)
            
            while not stop_event.is_set():
                publish_start_time = time.time()
                
                # 检测停止条件：action_stage_queues 为空且已经生成过动作
                if len(action_stage_queues) == 0:
                    log_message("发布线程检测到 action_stage_queues 为空，发出停止信号...", log_file)
                    stop_event.set()  # 由 publisher 发出停止信号
                    break
                
                # 从队列获取动作
                num_preds = len(action_stage_queues)
                weights = np.exp(-cfg.act_m * np.arange(num_preds))
                weights = weights / np.sum(weights)
                
                action = np.zeros(16)
                stage_id_arr = np.zeros(num_preds, dtype=int)
                for i in range(num_preds):
                    action_, stage_id_ = action_stage_queues[i].pop(0)
                    action += weights[i] * np.array(action_)
                    stage_id_arr[i] = stage_id_

                # 移除已空的动作队列
                while len(action_stage_queues) > 0 and len(action_stage_queues[0]) == 0:
                    action_stage_queues.pop(0)
                
                # 发布动作
                action = action.tolist()
                robot_interface.apply_action(action)

                # Voting + temporal integration in update_stage_progress; no extra pending/rollback layer
                if cfg.enable_stage_manager:
                    with stage_lock:
                        update_stage_progress(stage_progress, stage_id_arr, weights, cfg.mean_voting_flag)

                publish_count[0] += 1

                # 控制发布频率：30Hz
                publish_end_time = time.time()
                elapsed_interval = publish_end_time - publish_start_time
                if action_publish_interval > elapsed_interval:
                    time.sleep(action_publish_interval - elapsed_interval)
            
            # 发布线程完成时，等待同步停止
            log_message("发布线程准备停止，等待同步...", log_file)
            stop_barrier.wait()  # 等待两个线程都准备好后同时停止
                    
        except Exception as e:
            log_message(f"动作发布线程错误: {e}", log_file)
            exception_info[0] = e
            exception_occurred.set()
            stop_event.set()
            try:
                stop_barrier.wait(timeout=0.1)  # 尝试同步停止，但不要阻塞太久
            except threading.BrokenBarrierError:
                pass  # 如果另一个线程已经停止，忽略错误
    
    # 启动两个线程
    log_message("启动并行动作生成和发布线程...", log_file)
    generation_thread = threading.Thread(target=action_generation_thread, daemon=True)
    publishing_thread = threading.Thread(target=action_publishing_thread, daemon=True)
    
    generation_thread.start()
    publishing_thread.start()
    
    try:
        # 等待两个线程完成
        # 发布线程会在检测到 action_stage_queues 为空时设置 stop_event 并控制停止
        # 两个线程会通过 stop_barrier 同步停止
        # timeout 计算：生成阶段时间 + 发布剩余动作时间 + 缓冲
        # 最坏情况：生成线程停止时队列中有 max_episode_steps 个动作序列
        # 每个序列包含 num_action_horizon 个动作，需要以 control_frequency 发布
        generation_timeout = (
            cfg.max_episode_steps * action_generation_interval +  # 生成阶段时间
            (cfg.max_episode_steps * cfg.num_action_horizon) / cfg.control_frequency +  # 发布剩余动作时间
            1.0  # 缓冲时间
        )
        generation_thread.join(timeout=generation_timeout)
        publishing_thread.join(timeout=generation_timeout)
        
        if publishing_thread.is_alive():
            log_message("警告: 发布线程未在超时时间内停止，强制设置停止标志", log_file)
            stop_event.set()
            publishing_thread.join(timeout=0.5)
        
        # 检查是否有异常
        if exception_occurred.is_set():
            raise exception_info[0] if exception_info[0] else Exception("未知异常")
            
    except Exception as e:
        log_message(f"Episode 错误: {e}", log_file)
        stop_event.set()  # 确保线程停止
        raise
    finally:
        # 确保线程已停止（无论 try 块是否成功）
        # 如果线程在 try 块中已经完成，join() 会立即返回（不等待）
        # 如果线程还在运行，给它们 0.5 秒时间完成清理
        stop_event.set()
        if generation_thread.is_alive():
            generation_thread.join(timeout=0.5)
        if publishing_thread.is_alive():
            publishing_thread.join(timeout=0.5)
        log_message("所有线程已停止", log_file)



def run_eval_runtime(
    cfg: GenerateConfig,
    robot_interface: TeleavatarRobotInterface,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
    stage_classifier=None,
):
    """Run teleavatar runtime for multi episodes."""
    inference_time_ls = list()

    # Start Episodes
    for episode_idx in tqdm.tqdm(range(cfg.num_episodes)):
        log_message(f"Episode: {episode_idx}", log_file)

        # Run episode
        run_episode(
            cfg,
            robot_interface,
            model,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
            inference_time_ls,
            stage_classifier=stage_classifier,
        )

        log_message(f"Episode: {episode_idx} has finished!!", log_file)

        # Wait for scene reset before starting next episode
        if episode_idx < cfg.num_episodes - 1:
            reset_wait = 15
            log_message(f"Waiting {reset_wait}s for scene reset...", log_file)
            time.sleep(reset_wait)

    return inference_time_ls



@draccus.wrap()
def eval_teleavatar(cfg: GenerateConfig):
    """Main function to evaluate a trained policy on Teleavatar."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file = setup_logging(cfg)

    # Initialize model and components
    log_message("Initializing Finetuned VLA-Adapter...", log_file)
    start_time = time.time()
    model, action_head, proprio_projector, noisy_action_projector, processor, stage_classifier = initialize_model(cfg, log_file)
    initialize_model_period = time.time() - start_time
    log_message(f"Initialize model period: {initialize_model_period:.2f} seconds", log_file)
    log_message(f"model is on {model.device}", log_file)

    # Robot interface can initialize ros2_interface
    log_message("Initializing Teleavatar Robot Interface...", log_file)
    robot_interface = TeleavatarRobotInterface()

    # Start evaluation
    log_message("Starting Evaluation...", log_file)
    inference_time_ls = run_eval_runtime(
        cfg,
        robot_interface,
        model,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        log_file,
        stage_classifier=stage_classifier,
    )

    # Log final results
    log_message(f"Total episodes: {cfg.num_episodes}", log_file)
    log_message(f"Inference time list: {inference_time_ls}", log_file)
    # log_message(f"Max inference time: {max(inference_time_ls):.3f} seconds", log_file)
    log_message(f"Average inference time: {(sum(inference_time_ls) - max(inference_time_ls)) / (len(inference_time_ls)-1):.3f} seconds", log_file)

    # Gracefully shutdown ROS2 interface
    log_message("Shutting down ROS2 interface...", log_file)
    try:
        robot_interface.shutdown()
    except Exception as e:
        log_message(f"Error during ROS2 shutdown: {e}", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return


if __name__ == "__main__":
    eval_teleavatar()
