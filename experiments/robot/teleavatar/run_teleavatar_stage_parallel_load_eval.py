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
import queue  # For thread-safe queue
import threading  # For multi-threading

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
                                                     # max_episode_steps = 600 means one episode runs for about 1 minute

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
                                                     # Only models after vla-adapter-stage4 use this architecture

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
        # Fully parallel loading: main model, proprio_projector, action_head, processor, stage_classifier loaded simultaneously
        # Using ThreadPoolExecutor for parallel loading
        # Must disable low_cpu_mem_usage for parallel loading, otherwise the main model will use meta device for tensors,
        # causing action_head and other modules created in other threads to also become meta tensors, leading to .to(device) errors
        cfg.low_cpu_mem_usage = False
        max_workers = 1  # llm
        futures = {}

        # Calculate the number of threads needed
        if cfg.model_family == "openvla":
            max_workers += 1  # processor
        if cfg.use_proprio:
            max_workers += 1  # proprio_projector
        if cfg.use_l1_regression:
            max_workers += 1  # action_head
        max_workers += 1  # stage_classifier (vla-adapter-stage, loaded if exists in checkpoint)

        log_message(f"Starting parallel loading with {max_workers} threads", log_file)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit main model loading task
            futures[executor.submit(get_model, cfg)] = "model"
            log_message("Submitted llm loading task", log_file)

            # Submit processor loading task (if using openvla)
            processor = None
            if cfg.model_family == "openvla":
                futures[executor.submit(get_processor, cfg)] = "processor"
                log_message("Submitted processor loading task", log_file)

            # Submit proprio_projector loading task (if enabled)
            proprio_projector = None
            if cfg.use_proprio:
                futures[
                    executor.submit(
                        get_proprio_projector,
                        cfg,
                        cfg.llm_dim,
                        PROPRIO_DIM,  # 14-dimensional proprio for Teleavatar
                    )
                ] = "proprio"
                log_message("Submitted proprio_projector loading task", log_file)

            # Submit action_head loading task (if enabled)
            action_head = None
            if cfg.use_l1_regression:
                futures[
                    executor.submit(
                        get_action_head,
                        cfg,
                        cfg.llm_dim,
                    )
                ] = "action_head"
                log_message("Submitted action_head loading task", log_file)

            # Submit stage_classifier loading task (vla-adapter-stage, loaded from run_* directory)
            stage_classifier = None
            futures[executor.submit(get_stage_classifier, cfg, cfg.llm_dim)] = "stage_classifier"
            log_message("Submitted stage_classifier loading task", log_file)

            # Wait for all tasks to complete and collect results
            completed_count = 0
            model = None  # Initialize model variable

            for future in as_completed(futures):
                component_type = futures[future]
                try:
                    result = future.result()
                    completed_count += 1
                    
                    if   component_type == 'model':
                        model = result
                        # After main model loading completes, set version and check unnorm_key
                        model.set_version(cfg.save_version)
                        if cfg.model_family == "openvla":
                            check_unnorm_key(cfg, model)
                        log_message(f"Main model loaded ({completed_count}/{len(futures)})", log_file)
                    elif component_type == "processor":
                        processor = result
                        log_message(f"processor loaded ({completed_count}/{len(futures)})", log_file)
                    elif component_type == "proprio":
                        proprio_projector = result
                        log_message(f"proprio_projector loaded ({completed_count}/{len(futures)})", log_file)
                    elif component_type == "action_head":
                        action_head = result
                        log_message(f"action_head loaded ({completed_count}/{len(futures)})", log_file)
                    elif component_type == "stage_classifier":
                        stage_classifier = result
                        if stage_classifier is not None:
                            log_message(
                                f"stage_classifier (vla-adapter-stage) loaded ({completed_count}/{len(futures)})",
                                log_file,
                            )
                        else:
                            log_message(
                                f"stage_classifier weights not found, skipping ({completed_count}/{len(futures)})",
                                log_file,
                            )
                except Exception as e:
                    log_message(f"Error loading {component_type}: {e}", log_file)
                    raise

            # Verify main model loaded successfully
            if model is None:
                raise RuntimeError("Main model loading failed!")

            log_message(f"All components loaded! Total {len(futures)} components", log_file)

        return model, action_head, proprio_projector, noisy_action_projector, processor, stage_classifier
    else:
        # Sequential loading (original way)
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
            log_message("stage_classifier (vla-adapter-stage) loaded", log_file)
        else:
            log_message("stage_classifier weights not found, skipping", log_file)

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
    Use voting mechanism to update stage_progress
    Each Task goes through three states: "Waiting", "Active", "Done"
    Only one Task can be in Active state at any time
    """
    # Weighted voting
    if mean_voting_flag is not True:
        weighted_score = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for idx, stage_id in enumerate(stage_id_arr):
            weighted_score[stage_id] += weights[idx]

        # "Effective stage" is the stage_id with highest weighted score
        effective_stage = max(weighted_score, key=weighted_score.get)
        if stage_progress[effective_stage] == "Waiting":
            # Current task needs to start processing, close other active tasks
            for stage_id, stage_state in stage_progress.items():
                if stage_state == "Active":
                    stage_progress[stage_id] = "Done"
                    break
            stage_progress[effective_stage] = "Active"
        elif stage_progress[effective_stage] == "Active":
            # Current task is being processed, no update needed
            pass
        elif stage_progress[effective_stage] == "Done":
            # Current stage was already completed before, no update needed
            pass
        return stage_progress

    # Mean voting
    else:
        weighted_score = {1: 0, 2: 0, 3: 0, 4: 0}
        for stage_id in stage_id_arr:
            weighted_score[stage_id] += 1

        # Find all stage_ids with score equal to the highest score
        max_score = max(weighted_score.values())
        best_stage_ids = [stage_id for stage_id, score in weighted_score.items() if score == max_score]

        # First remove stage_ids that are Done
        best_stage_ids = list(filter(lambda stage_id: stage_progress[stage_id] != "Done", best_stage_ids))
        if best_stage_ids == []:
            return stage_progress

        # Then check if any is Active, if so no change needed
        for stage_id in best_stage_ids:
            if stage_progress[stage_id] == "Active":
                return stage_progress

        # If no Active, all stage_ids are waiting, randomly choose one as Active
        effective_stage = random.choice(best_stage_ids)
        # Current task needs to start processing, close other active tasks
        for stage_id, stage_state in stage_progress.items():
            if stage_state == "Active":
                stage_progress[stage_id] = "Done"
                break
        stage_progress[effective_stage] = "Active"
        return stage_progress

    # Mean voting
    else:
        weighted_score = {1: 0, 2: 0, 3: 0, 4: 0}
        for stage_id in stage_id_arr:
            weighted_score[stage_id] += 1

        # Find all stage_ids with score equal to the highest score
        max_score = max(weighted_score.values())
        best_stage_ids = [stage_id for stage_id, score in weighted_score.items() if score == max_score]

        # First remove stage_ids that are Done
        best_stage_ids = list(filter(lambda stage_id: stage_progress[stage_id] != "Done", best_stage_ids))
        if best_stage_ids == []:
            return stage_progress

        # Then check if any is Active, if so no change needed
        for stage_id in best_stage_ids:
            if stage_progress[stage_id] == "Active":
                return stage_progress

        # If no Active, all stage_ids are waiting, randomly choose one as Active
        effective_stage = random.choice(best_stage_ids)
        # Current task needs to start processing, close other active tasks
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
    # Stores multiple action queues, but only the first nine or fewer participate in action generation at current time
    action_stage_queues = list()
    # Records the progress of four stages in one episode
    stage_progress = {1: "Waiting", 2: "Waiting", 3: "Waiting", 4: "Waiting"}

    # Thread stop event flags
    stop_event = threading.Event()
    exception_occurred = threading.Event()
    exception_info = [None]  # Used to store exception information

    # Create synchronization barriers to ensure both threads start/stop simultaneously (requires 2 threads)
    start_barrier = threading.Barrier(2)
    stop_barrier = threading.Barrier(2)

    # Action generation frequency: 10Hz = 0.1 second interval
    action_generation_interval = 1.0 / cfg.action_generation_frequency  # Using configured frequency
    # Action publishing frequency: 30Hz = 0.033 second interval
    action_publish_interval = 1.0 / cfg.control_frequency  # 30Hz

    inference_count = [0]  # Use list for sharing between threads
    publish_count = [0]  # Publishing steps; log prefixes [GEN]/[STAGE]/[PUB] for easy grep filtering

    # stage_progress is shared between generation & publishing threads (stage_lock)
    stage_log_buffer = []
    stage_log_last_flush = [time.time()]
    stage_lock = threading.Lock()

    def action_generation_thread():
        """Thread that generates actions at 10Hz and puts them into queue"""
        try:
            start_barrier.wait()  # Wait for both threads to be ready before starting together

            # Generation thread only stops when stop_event is set, controlled by publishing thread
            # Safety check: if max steps reached, log warning but continue (publishing thread decides when to stop)
            print_generation_stop_flag = False

            # Write subtask content into the prompt
            task_description = prepare_task_description(cfg)

            while not stop_event.is_set():
                # Safety check: if max steps reached, wait for stop signal from publishing thread
                if inference_count[0] >= cfg.max_episode_steps:
                    if print_generation_stop_flag is False:
                        log_message(
                            f"Warning: reached max steps {cfg.max_episode_steps}, but continuing to wait for publishing thread stop signal", log_file
                        )
                        print_generation_stop_flag = True
                    time.sleep(action_publish_interval)
                    continue

                # Normal action generation execution
                generation_start_time = time.time()

                # Convert stage_progress to low_level_task
                if cfg.enable_stage_manager:
                    with stage_lock:
                        low_level_task = ",".join(
                            f"Task{sid}:{stage_progress[sid]}" for sid in range(1, len(stage_progress) + 1)
                        )
                else:
                    low_level_task = "Task1:Waiting,Task2:Waiting,Task3:Waiting,Task4:Waiting"

                # Get observation
                obs = robot_interface.get_observation()
                observation = prepare_observation(obs)

                # Generate actions (also returns current stage id 0/1/2/3 if stage_classifier is loaded)
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

                # Log image patch count and prompt token count
                log_message(
                    f"num_patches: {num_patches}, num_prompt_tokens: {num_prompt_tokens}",
                    log_file,
                )

                # Put (action, stage_id) sequence into action queue
                max_index, _ = max(enumerate(stage_probs), key=lambda x: x[1])
                stage_id = max_index + 1  # Convert idx to stage_id
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

                # Control generation frequency: 10Hz
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

            # Generation thread detects stop_event (set by publishing thread), wait for synchronized stop
            log_message("Generation thread detected stop signal, preparing to stop...", log_file)
            stop_barrier.wait()  # Wait for both threads to be ready before stopping together

        except Exception as e:
            log_message(f"Action generation thread error: {e}", log_file)
            exception_info[0] = e
            exception_occurred.set()
            stop_event.set()
            try:
                stop_barrier.wait(timeout=0.1)  # Try to synchronize stop, but don't block too long
            except threading.BrokenBarrierError:
                pass  # If another thread has already stopped, ignore error

    def action_publishing_thread():
        """Thread that takes actions from queue and publishes at 30Hz"""
        try:
            start_barrier.wait()  # Wait for both threads to be ready before starting together

            # Wait for model to generate first action
            while len(action_stage_queues) == 0:
                time.sleep(action_publish_interval)
            
            while not stop_event.is_set():
                publish_start_time = time.time()

                # Check stop condition: action_stage_queues is empty and actions have been generated
                if len(action_stage_queues) == 0:
                    log_message("Publishing thread detected action_stage_queues is empty, sending stop signal...", log_file)
                    stop_event.set()  # Stop signal sent by publisher
                    break

                # Get actions from queue
                num_preds = len(action_stage_queues)
                weights = np.exp(-cfg.act_m * np.arange(num_preds))
                weights = weights / np.sum(weights)
                
                action = np.zeros(16)
                stage_id_arr = np.zeros(num_preds, dtype=int)
                for i in range(num_preds):
                    action_, stage_id_ = action_stage_queues[i].pop(0)
                    action += weights[i] * np.array(action_)
                    stage_id_arr[i] = stage_id_

                # Remove empty action queues
                while len(action_stage_queues) > 0 and len(action_stage_queues[0]) == 0:
                    action_stage_queues.pop(0)

                # Publish action
                action = action.tolist()
                robot_interface.apply_action(action)

                # Voting + temporal integration in update_stage_progress; no extra pending/rollback layer
                if cfg.enable_stage_manager:
                    with stage_lock:
                        update_stage_progress(stage_progress, stage_id_arr, weights, cfg.mean_voting_flag)

                publish_count[0] += 1

                # Control publishing frequency: 30Hz
                publish_end_time = time.time()
                elapsed_interval = publish_end_time - publish_start_time
                if action_publish_interval > elapsed_interval:
                    time.sleep(action_publish_interval - elapsed_interval)

            # When publishing thread completes, wait for synchronized stop
            log_message("Publishing thread preparing to stop, waiting for synchronization...", log_file)
            stop_barrier.wait()  # Wait for both threads to be ready before stopping together

        except Exception as e:
            log_message(f"Action publishing thread error: {e}", log_file)
            exception_info[0] = e
            exception_occurred.set()
            stop_event.set()
            try:
                stop_barrier.wait(timeout=0.1)  # Try to synchronize stop, but don't block too long
            except threading.BrokenBarrierError:
                pass  # If another thread has already stopped, ignore error

    # Start two threads
    log_message("Starting parallel action generation and publishing threads...", log_file)
    generation_thread = threading.Thread(target=action_generation_thread, daemon=True)
    publishing_thread = threading.Thread(target=action_publishing_thread, daemon=True)

    generation_thread.start()
    publishing_thread.start()

    try:
        # Wait for both threads to complete
        # Publishing thread will set stop_event when action_stage_queues is empty and control stopping
        # Both threads will synchronize stop via stop_barrier
        # timeout calculation: generation phase time + remaining action publishing time + buffer
        # Worst case: when generation thread stops, queue has max_episode_steps action sequences
        # Each sequence contains num_action_horizon actions, need to be published at control_frequency
        generation_timeout = (
            cfg.max_episode_steps * action_generation_interval  # Generation phase time
            + (cfg.max_episode_steps * cfg.num_action_horizon) / cfg.control_frequency  # Remaining action publishing time
            + 1.0  # Buffer time
        )
        generation_thread.join(timeout=generation_timeout)
        publishing_thread.join(timeout=generation_timeout)
        
        if publishing_thread.is_alive():
            log_message("Warning: publishing thread did not stop within timeout, forcing stop signal", log_file)
            stop_event.set()
            publishing_thread.join(timeout=0.5)

        # Check for exceptions
        if exception_occurred.is_set():
            raise exception_info[0] if exception_info[0] else Exception("Unknown exception")

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        stop_event.set()  # Ensure threads stop
        raise
    finally:
        # Ensure threads have stopped (whether try block succeeded or not)
        # If threads already completed in try block, join() returns immediately (no wait)
        # If threads still running, give them 0.5 seconds to complete cleanup
        stop_event.set()
        if generation_thread.is_alive():
            generation_thread.join(timeout=0.5)
        if publishing_thread.is_alive():
            publishing_thread.join(timeout=0.5)
        log_message("All threads stopped", log_file)


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
