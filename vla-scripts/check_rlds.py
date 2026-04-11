"""
Read two RLDS format datasets
"""

from pathlib import Path
from typing import Dict, Any, Optional
import tensorflow as tf
import dlimp as dl

from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform, teleavatar_dataset_transform
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType


def load_libero_dataset(data_root_dir: Path, train: bool = True):
    """
    Load LIBERO dataset

    Args:
        data_root_dir: Data root directory, e.g., Path("/home/nas/VLA-Adapter/data")
        train: Whether to use training set

    Returns:
        dataset: RLDS dataset
        dataset_statistics: Dataset statistics information
    """
    dataset_name = "libero_spatial_no_noops"
    data_dir = str(data_root_dir / "libero")

    # LIBERO dataset configuration (from configs.py)
    dataset_configs = OXE_DATASET_CONFIGS.get(dataset_name)
    image_obs_keys = dataset_configs["image_obs_keys"]
    depth_obs_keys = dataset_configs["depth_obs_keys"]
    state_obs_keys = dataset_configs["state_obs_keys"]
    language_key = "language_instruction"

    # Use LIBERO standardization function
    standardize_fn = libero_dataset_transform

    # Use BOUNDS_Q99 normalization (LIBERO's default setting)
    normalization_type = NormalizationType.BOUNDS_Q99
    
    dataset, dataset_statistics = make_dataset_from_rlds(
        name=dataset_name,
        data_dir=data_dir,
        train=train,
        standardize_fn=standardize_fn,
        shuffle=True,
        image_obs_keys=image_obs_keys,
        depth_obs_keys=depth_obs_keys,
        state_obs_keys=state_obs_keys,
        language_key=language_key,
        action_proprio_normalization_type=normalization_type,
        dataset_statistics=None,  # Auto-compute statistics
        num_parallel_reads=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset, dataset_statistics


def load_shihaoran_dataset(data_root_dir: Path, train: bool = True):
    """
    Load Shihaoran dataset

    Args:
        data_root_dir: Data root directory, e.g., Path("/home/nas/VLA-Adapter/data")
        train: Whether to use training set

    Returns:
        dataset: RLDS dataset
        dataset_statistics: Dataset statistics information
    """
    dataset_name = "right_grip_grab_a_stuffed_animal_into_left_box"
    data_dir = str(data_root_dir / "shihaoran")

    # Note: These configurations may need adjustment based on actual dataset structure
    # If dataset structure is similar to LIBERO, can use the following configurations
    # If different, need to adjust based on actual observation and action key names
    dataset_configs = OXE_DATASET_CONFIGS.get(dataset_name)
    image_obs_keys = dataset_configs["image_obs_keys"]
    depth_obs_keys = dataset_configs["depth_obs_keys"]
    state_obs_keys = dataset_configs["state_obs_keys"]
    language_key = "language_instruction"  # If dataset has language instruction

    # If no standardization function, can set to None or customize
    standardize_fn = teleavatar_dataset_transform

    # Use BOUNDS_Q99 normalization
    normalization_type = NormalizationType.BOUNDS_Q99
    
    dataset, dataset_statistics = make_dataset_from_rlds(
        name=dataset_name,
        data_dir=data_dir,
        train=train,
        standardize_fn=standardize_fn,
        shuffle=True,
        image_obs_keys=image_obs_keys,
        depth_obs_keys=depth_obs_keys,
        state_obs_keys=state_obs_keys,
        language_key=language_key,
        action_proprio_normalization_type=normalization_type,
        dataset_statistics=None,  # Auto-compute statistics
        num_parallel_reads=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset, dataset_statistics


def inspect_dataset_structure(dataset, num_samples: int = 1):
    """
    Inspect dataset structure, print information of first few samples

    Args:
        dataset: RLDS dataset
        num_samples: Number of samples to inspect
    """
    print("=" * 80)
    print("Dataset Structure Inspection")
    print("=" * 80)
    
    iterator = dataset.iterator()
    for i, sample in enumerate(iterator):
        if i >= num_samples:
            break

        print(f"\nSample {i + 1}:")
        print(f"  Keys: {list(sample.keys())}")

        for key, value in sample.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'shape'):
                        print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                    elif isinstance(sub_value, tf.Tensor):
                        print(f"    {sub_key}: Tensor, dtype={sub_value.dtype}")
                    else:
                        print(f"    {sub_key}: {type(sub_value)}")
            elif hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    
    print("=" * 80)


def main():
    """Main function: read and inspect two datasets"""
    # Set data root directory
    data_root_dir = Path("/home/nas/VLA-Adapter/data")

    print("Loading LIBERO dataset...")
    try:
        libero_dataset, libero_stats = load_libero_dataset(data_root_dir, train=True)
        print(f"✓ LIBERO dataset loaded successfully")
        print(f"  Dataset statistics: {list(libero_stats.keys())}")
        print(f"  Number of trajectories: {libero_stats.get('num_trajectories', 'N/A')}")
        print(f"  Number of transitions: {libero_stats.get('num_transitions', 'N/A')}")

        # Inspect dataset structure
        print("\nInspecting LIBERO dataset structure:")
        inspect_dataset_structure(libero_dataset, num_samples=1)

    except Exception as e:
        print(f"✗ LIBERO dataset loading failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")

    print("Loading Shihaoran dataset...")
    try:
        shihaoran_dataset, shihaoran_stats = load_shihaoran_dataset(data_root_dir, train=True)
        print(f"✓ Shihaoran dataset loaded successfully")
        print(f"  Dataset statistics: {list(shihaoran_stats.keys())}")
        print(f"  Number of trajectories: {shihaoran_stats.get('num_trajectories', 'N/A')}")
        print(f"  Number of transitions: {shihaoran_stats.get('num_transitions', 'N/A')}")

        # Inspect dataset structure
        print("\nInspecting Shihaoran dataset structure:")
        inspect_dataset_structure(shihaoran_dataset, num_samples=1)

    except Exception as e:
        print(f"✗ Shihaoran dataset loading failed: {e}")
        print(
            "\nHint: If loading fails, you may need to adjust the following parameters based on actual dataset structure:"
        )
        print("  - image_obs_keys: Image observation key names")
        print("  - state_obs_keys: State observation key names")
        print("  - language_key: Language instruction key name (if exists)")
        print("  - standardize_fn: Data standardization function (if needed)")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Dataset loading completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()