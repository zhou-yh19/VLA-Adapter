"""
读取两个 RLDS 格式的数据集
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
    加载 LIBERO 数据集
    
    Args:
        data_root_dir: 数据根目录，例如 Path("/home/nas/VLA-Adapter/data")
        train: 是否使用训练集
    
    Returns:
        dataset: RLDS 数据集
        dataset_statistics: 数据集统计信息
    """
    dataset_name = "libero_spatial_no_noops"
    data_dir = str(data_root_dir / "libero")
    
    # LIBERO 数据集的配置（从 configs.py 中获取）
    dataset_configs = OXE_DATASET_CONFIGS.get(dataset_name)
    image_obs_keys = dataset_configs['image_obs_keys']
    depth_obs_keys = dataset_configs['depth_obs_keys']
    state_obs_keys = dataset_configs['state_obs_keys']
    language_key = "language_instruction"
    
    # 使用 LIBERO 的标准化函数
    standardize_fn = libero_dataset_transform
    
    # 使用 BOUNDS_Q99 归一化（LIBERO 的默认设置）
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
        dataset_statistics=None,  # 自动计算统计信息
        num_parallel_reads=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset, dataset_statistics


def load_shihaoran_dataset(data_root_dir: Path, train: bool = True):
    """
    加载 Shihaoran 数据集
    
    Args:
        data_root_dir: 数据根目录，例如 Path("/home/nas/VLA-Adapter/data")
        train: 是否使用训练集
    
    Returns:
        dataset: RLDS 数据集
        dataset_statistics: 数据集统计信息
    """
    dataset_name = "right_grip_grab_a_stuffed_animal_into_left_box"
    data_dir = str(data_root_dir / "shihaoran")
    
    # 注意：这些配置可能需要根据实际数据集结构调整
    # 如果数据集结构与 LIBERO 类似，可以使用以下配置
    # 如果不同，需要根据实际 observation 和 action 的键名调整
    dataset_configs = OXE_DATASET_CONFIGS.get(dataset_name)
    image_obs_keys = dataset_configs['image_obs_keys']
    depth_obs_keys = dataset_configs['depth_obs_keys']
    state_obs_keys = dataset_configs['state_obs_keys']
    language_key = "language_instruction"  # 如果数据集有语言指令
    
    # 如果没有标准化函数，可以设置为 None 或自定义
    standardize_fn = teleavatar_dataset_transform
    
    # 使用 BOUNDS_Q99 归一化
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
        dataset_statistics=None,  # 自动计算统计信息
        num_parallel_reads=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset, dataset_statistics


def inspect_dataset_structure(dataset, num_samples: int = 1):
    """
    检查数据集结构，打印前几个样本的信息
    
    Args:
        dataset: RLDS 数据集
        num_samples: 要检查的样本数量
    """
    print("=" * 80)
    print("数据集结构检查")
    print("=" * 80)
    
    iterator = dataset.iterator()
    for i, sample in enumerate(iterator):
        if i >= num_samples:
            break
        
        print(f"\n样本 {i+1}:")
        print(f"  键: {list(sample.keys())}")
        
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
    """主函数：读取并检查两个数据集"""
    # 设置数据根目录
    data_root_dir = Path("/home/nas/VLA-Adapter/data")
    
    print("正在加载 LIBERO 数据集...")
    try:
        libero_dataset, libero_stats = load_libero_dataset(data_root_dir, train=True)
        print(f"✓ LIBERO 数据集加载成功")
        print(f"  数据集统计信息: {list(libero_stats.keys())}")
        print(f"  轨迹数: {libero_stats.get('num_trajectories', 'N/A')}")
        print(f"  转换数: {libero_stats.get('num_transitions', 'N/A')}")
        
        # 检查数据集结构
        print("\n检查 LIBERO 数据集结构:")
        inspect_dataset_structure(libero_dataset, num_samples=1)
        
    except Exception as e:
        print(f"✗ LIBERO 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80 + "\n")
    
    print("正在加载 Shihaoran 数据集...")
    try:
        shihaoran_dataset, shihaoran_stats = load_shihaoran_dataset(data_root_dir, train=True)
        print(f"✓ Shihaoran 数据集加载成功")
        print(f"  数据集统计信息: {list(shihaoran_stats.keys())}")
        print(f"  轨迹数: {shihaoran_stats.get('num_trajectories', 'N/A')}")
        print(f"  转换数: {shihaoran_stats.get('num_transitions', 'N/A')}")
        
        # 检查数据集结构
        print("\n检查 Shihaoran 数据集结构:")
        inspect_dataset_structure(shihaoran_dataset, num_samples=1)
        
    except Exception as e:
        print(f"✗ Shihaoran 数据集加载失败: {e}")
        print("\n提示：如果加载失败，可能需要根据实际数据集结构调整以下参数：")
        print("  - image_obs_keys: 图像观察的键名")
        print("  - state_obs_keys: 状态观察的键名")
        print("  - language_key: 语言指令的键名（如果存在）")
        print("  - standardize_fn: 数据标准化函数（如果需要）")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("数据集加载完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()