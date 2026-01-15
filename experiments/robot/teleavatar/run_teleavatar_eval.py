"""
run_teleavatar_eval.py

Evaluates a trained policy on a Teleavatar.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import time

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
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
    num_open_loop_steps: int = 10                    # Number of actions to execute before querying policy again
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Teleavatar runtime parameters
    #################################################################################################################
    control_frequency: float = 30                    # Control loop frequency in Hz               
    task_description: str = "right_grip_grab_a_stuffed_animal_into_left_box" 
                                                     # Language instruction for the robot
    num_episodes: int = 10                           # Number of episodes to run
    max_episode_steps: int = 90                     # Maximum VLA inference count per episode (0 = unlimited)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter-teleavatar"     # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=PROPRIO_DIM,  # 14-dimensional proprio for Teleavatar
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


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



def run_episode(
    cfg: GenerateConfig,
    task_description: str,
    robot_interface: TeleavatarRobotInterface,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Run episode
    try:
        t = 0
        action_publish_interval = 1.0 / cfg.control_frequency
        start_time = None

        # VLA inference count is cfg.max_episode_steps
        while t < cfg.max_episode_steps:
            
            if len(action_queue) == 0:
                # If action queue is empty, requery model
                obs = robot_interface.get_observation()
                observation = prepare_observation(obs)

                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )

                # Add actions to queue
                action_queue.extend(actions)
                t += 1

            # Get action from queue
            action = action_queue.popleft()

            # publish action to Teleavatar via ROS2 interface
            if start_time is not None:
                stop_time = time.time()
                interval = stop_time - start_time
                if interval < action_publish_interval:
                    time.sleep(action_publish_interval - interval)
            robot_interface.apply_action(action)
            start_time = time.time()

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)



def run_eval_runtime(
    cfg: GenerateConfig,
    robot_interface: TeleavatarRobotInterface,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run teleavatar runtime for multi episodes."""
    # Start Episodes
    for episode_idx in tqdm.tqdm(range(cfg.num_episodes)):
        log_message(f"\nEpisode: {episode_idx}", log_file)

        # Run episode
        run_episode(
            cfg,
            cfg.task_description.replace("_", " "),
            robot_interface,
            model,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )

        log_message(f"\nEpisode: {episode_idx} has finished!!", log_file)



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
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    initialize_model_period = time.time() - start_time
    log_message(f"Initialize model period: {initialize_model_period:.2f} seconds", log_file)
    log_message(f"model is on {model.device}", log_file)

    # Robot interface can initialize ros2_interface
    log_message("Initializing Teleavatar Robot Interface...", log_file)
    robot_interface = TeleavatarRobotInterface()

    # Start evaluation
    log_message("Starting Evaluation...", log_file)
    run_eval_runtime(
        cfg,
        robot_interface,
        model,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        log_file,
    )

    # Log final results
    log_message(f"Total episodes: {cfg.num_episodes}", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return


if __name__ == "__main__":
    eval_teleavatar()
