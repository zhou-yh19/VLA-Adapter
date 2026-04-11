"""Utils for evaluating VLA-Adapter or fine-tuned VLA-Adapter policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_TOKENS,
    NUM_STAGES,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def find_checkpoint_file_optional(pretrained_checkpoint: str, file_pattern: str) -> Optional[str]:
    """
    Find a checkpoint file matching a pattern if it exists.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        Optional[str]: Path to the matching checkpoint file, or None if not found or not a directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return None
    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)
    if len(checkpoint_files) == 0:
        return None
    # Prefer "latest" if multiple (e.g. stage_classifier--latest_checkpoint.pt vs stage_classifier--20000_checkpoint.pt)
    for p in checkpoint_files:
        if "latest" in p:
            return p
    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

def load_component_state_dict_v1(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v

    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # Load the model (must be False for parallel loading, otherwise meta tensor will contaminate modules created in other threads)
    low_cpu = getattr(cfg, "low_cpu_mem_usage", True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=low_cpu,
        trust_remote_code=False,
    )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    """
    Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.

    Args:
        vla: The VLA model
        cfg: Configuration object with model parameters

    Returns:
        torch.nn.Module: VLA model with FiLM applied
    """
    from peft import LoraConfig, get_peft_model

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # Create and apply FiLMed vision backbone
    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    # Load vision backbone checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    # Use the model component instead of wrapper and convert to bfloat16
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    # Detect if it's a local path: if path exists and is a directory, treat as local path
    # This avoids delays and potential network issues from calling HfApi
    is_local_path = os.path.isdir(cfg.pretrained_checkpoint) or os.path.exists(cfg.pretrained_checkpoint)

    # If it looks like a local path, use trust_remote_code=True directly
    # Otherwise try to detect if it's an HF Hub path
    if is_local_path:
        trust_remote_code = True
    else:
        # Try to detect if it's an HF Hub path
        trust_remote_code = not model_is_on_hf_hub(cfg.pretrained_checkpoint)

    # If it's a local checkpoint, register Auto Classes first (consistent with get_vla)
    if trust_remote_code:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=trust_remote_code)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data

    Returns:
        ProprioProjector: The initialized proprio projector
    """
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "VLA-Adapter/LIBERO-Spatial-Pro": "proprio_projector--checkpoint.pt",
            "VLA-Adapter/LIBERO-Object-Pro": "proprio_projector--checkpoint.pt",
            "VLA-Adapter/LIBERO-Goal-Pro": "proprio_projector--checkpoint.pt",
            "VLA-Adapter/LIBERO-Long-Pro": "proprio_projector--checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    """
    Get noisy action projector for diffusion-based action prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        NoisyActionProjector: The initialized noisy action projector
    """
    # Initialize projector and move to device
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)

    return noisy_action_projector


def get_action_head(cfg: Any, llm_dim: int) -> Union[L1RegressionActionHead]:
    """
    Get action head for continuous value prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

    Raises:
        AssertionError: If both L1 regression and diffusion are specified
    """
    # assert not cfg.use_l1_regression, "Cannot use both L1 regression and diffusion action head!"
    if not hasattr(cfg, "use_pro_version"):
        if "Pro" in cfg.pretrained_checkpoint or "Pro" in cfg.save_version:
            cfg.use_pro_version = True
        else:
            cfg.use_pro_version = False

    # Initialize appropriate action head based on configuration
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=llm_dim, 
            hidden_dim=llm_dim, 
            action_dim=ACTION_DIM,
            use_pro_version=cfg.use_pro_version,
        )

    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "VLA-Adapter/LIBERO-Spatial-Pro": "action_head--checkpoint.pt",
            "VLA-Adapter/LIBERO-Object-Pro": "action_head--checkpoint.pt",
            "VLA-Adapter/LIBERO-Goal-Pro": "action_head--checkpoint.pt",
            "VLA-Adapter/LIBERO-Long-Pro": "action_head--checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head


def get_stage_classifier(cfg: Any, llm_dim: int):
    """
    Load stage classifier (vla-adapter-stage) from checkpoint if present.

    The stage classifier is a small head that predicts stage from VLA stage tokens.
    Used when the checkpoint was trained with stage classification (e.g. finetune_stage_server.py).

    Auto-detects checkpoint format: StageClassifier_0 (W, w, b) vs StageClassifier_1 (fc1, fc2).

    Args:
        cfg: Configuration with pretrained_checkpoint path
        llm_dim: LLM hidden dimension (e.g. 896)

    Returns:
        Stage classifier module on DEVICE in eval mode, or None if no stage_classifier checkpoint found
    """
    from experiments.robot.teleavatar.stage_classifier import StageClassifier_0, StageClassifier_1
    from prismatic.vla.constants import NUM_STAGE_CLASSES

    checkpoint_path = find_checkpoint_file_optional(cfg.pretrained_checkpoint, "stage_classifier")
    if checkpoint_path is None:
        return None

    state_dict = load_component_state_dict(checkpoint_path)
    # Detect architecture: old checkpoints use (W, w, b), new use (fc1, fc2)
    if "W" in state_dict and "w" in state_dict and "b" in state_dict:
        stage_classifier = StageClassifier_0(
            llm_dim=llm_dim,
            num_stage_classes=NUM_STAGE_CLASSES,
        ).to(DEVICE)
    else:
        stage_classifier = StageClassifier_1(
            llm_dim=llm_dim,
            num_stage_classes=NUM_STAGE_CLASSES,
            hidden_dim=128,
            dropout_rate=0.5,
        ).to(DEVICE)
    stage_classifier.load_state_dict(state_dict, strict=True)
    stage_classifier.eval()
    return stage_classifier


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    use_minivlm: bool = False,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

        # Build VLA prompt
        if not use_minivlm:
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        else:
            prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n<|im_start|>assistant\n'

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]
                
        # Generate action
        if action_head is None:
            # Standard VLA output (single-image inputs, discrete actions)
            action, _, num_patches, num_prompt_tokens = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            # Custom action head for continuous actions
            action, _, num_patches, num_prompt_tokens = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
            )

    # Extract subset of actions for open loop steps
    return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]


def get_vla_stage_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    high_level_task: str,
    low_level_task: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    use_minivlm: bool = False,
    stage_classifier: Optional[torch.nn.Module] = None,
) -> Tuple[List[np.ndarray], Optional[int]]:
    """
    Generate action predictions and current stage id (0/1/2/3) with the VLA policy.

    When stage_classifier is not None, uses a forward hook on the VLA language model to
    capture the last-layer hidden states at stage token positions, then runs the stage
    classifier and returns argmax(logits) as the current stage id. Does not modify
    modeling_prismatic.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM
        stage_classifier: Optional stage classifier (StageClassifier_1). If provided,
            current stage id is computed and returned.

    Returns:
        Tuple of (list of predicted actions, stage_id or None).
        stage_id is in {0, 1, 2, 3} when stage_classifier is used, else None.
    """
    captured: Dict[str, Optional[torch.Tensor]] = {"hidden_states": None}

    def _capture_last_hidden(module: torch.nn.Module, input: Any, output: Any) -> None:
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            captured["hidden_states"] = output.hidden_states[-1]

    hook_handle = None
    if stage_classifier is not None:
        hook_handle = vla.language_model.register_forward_hook(_capture_last_hidden)

    with torch.inference_mode():
        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)
        
        # Build VLA prompt
        hl_token_count = None
        if not use_minivlm:
            prompt = f"In: What action should the robot take to {high_level_task.lower()} {low_level_task.lower()}?\nOut:"
        else:
            high_level_task_prompt = (
                f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\nWhat action should the robot take to {high_level_task.lower()}?<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            low_level_task_prompt = f'{low_level_task.lower()}'
            # prompt = f"{high_level_task_prompt}{low_level_task_prompt}<|im_end|>\n"
            prompt = f"{high_level_task_prompt}{low_level_task_prompt}"

            # When stage_insert_between_tasks is enabled, compute the token count of HL prompt,
            # used for inserting stage_queries between HL and LL
            stage_insert_between_tasks = getattr(cfg, "stage_insert_between_tasks", False)
            if stage_insert_between_tasks:
                _tokenizer = getattr(processor, "tokenizer", None)
                if _tokenizer is None:
                    raise AttributeError("processor.tokenizer is required for stage_insert_between_tasks")
                _hl_ids = _tokenizer(
                    high_level_task_prompt, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                hl_token_count = int(_hl_ids.shape[-1])

            # Calculate high_level_task tokens and low_level_task tokens. This is not commonly used
            # Only for small/temporary token counting tests: default is not executed, change if False to if True when needed.
            # Goal: compute the token count corresponding to high_level_task_prompt / low_level_task_prompt after converting to embedding.
            if False:
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise AttributeError("processor.tokenizer is required for token counting")

                hl_ids = tokenizer(
                    high_level_task_prompt, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                ll_ids = tokenizer(
                    low_level_task_prompt, return_tensors="pt", add_special_tokens=False
                )["input_ids"]

                hl_token_count = int(hl_ids.shape[-1])
                ll_token_count = int(ll_ids.shape[-1])
                # Number of embeddings matches number of tokens
                print(f"[language_token_count_debug] high_level_task_prompt: tokens={hl_token_count}")
                print(f"[language_token_count_debug] low_level_task_prompt: tokens={ll_token_count}")

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]
                
        # Generate action
        if action_head is None:
            # Standard VLA output (single-image inputs, discrete actions)
            action, _, num_patches, num_prompt_tokens = vla.predict_action(
                **inputs, unnorm_key=cfg.unnorm_key, do_sample=False,
                hl_token_count=hl_token_count,
            )
        else:
            # Custom action head for continuous actions
            action, _, num_patches, num_prompt_tokens = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
                hl_token_count=hl_token_count,
            )

    if hook_handle is not None:
        hook_handle.remove()

    actions = [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]

    if stage_classifier is not None and captured["hidden_states"] is not None:
        last_layer = captured["hidden_states"]
        if hl_token_count is not None:
            # New mode: stage_queries between HL and LL, with LL tokens after
            # End of sequence structure: ..., stage(8), ll_tokens(ll_full), action(64), stop(1)
            ll_full_count = num_prompt_tokens - hl_token_count
            tail_offset = NUM_TOKENS + 1 + ll_full_count
            stage_hidden = last_layer[
                :, -(tail_offset + NUM_STAGES) : -tail_offset, :
            ]
        else:
            # Old mode: stage_queries right before action tokens
            stage_hidden = last_layer[:, -(NUM_TOKENS + 1 + NUM_STAGES) : -(NUM_TOKENS + 1), :]
        stage_logits = stage_classifier(stage_hidden.float())

        # 1. Normalization: apply Softmax on the last dimension (dim=-1) to convert to probability distribution
        stage_probs = F.softmax(stage_logits, dim=-1)

        # 2. Reduce dimensions and convert:
        # - .squeeze() compresses shape from [1, 1, 4] to [4]
        # - .detach().cpu().tolist() safely converts to a Python 1D list
        stage_probs_ls = stage_probs.squeeze().detach().cpu().tolist()
        
        # stage_id = int(stage_logits.argmax(dim=-1).item())

    return actions, stage_probs_ls, num_patches, num_prompt_tokens


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
