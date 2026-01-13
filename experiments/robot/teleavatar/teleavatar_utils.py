"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import numpy as np
import tensorflow as tf

from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import logging

import rclpy
from rclpy.node import Node


##########################################################################################
# TeleAvatar Specific Utils
##########################################################################################

def get_teleavatar_chest_image(obs):
    """Get chest raw-image from ros2 interface."""
    return obs['images']['head_camera']


def get_teleavatar_left_wrist_image(obs):
    """Get left-wrist raw-image from ros2 interface."""
    return obs['images']['left_color']


def get_teleavatar_right_wrist_image(obs):
    """Get right-wrist raw-image from ros2 interface."""
    return obs['images']['right_color']


def get_teleavatar_state(obs, norm_stats):
    """Get proprio from ros2 interface and normalize"""
    state_data = obs['state']
    # Normalize proprio
    return normalize_proprio_for_teleavatar(state_data, norm_stats)


def normalize_proprio_for_teleavatar(
    proprio: np.ndarray, 
    norm_stats: Dict[str, Any]
) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution for TeleAvatar.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics. TeleAvatar use NormalizationType.BOUNDS_Q99

    Returns:
        np.ndarray: Normalized proprioception data
    """
    mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
    proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])

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


def denormalize_action_for_teleavatar(
    normalized_action: np.ndarray, 
    norm_stats: Dict[str, Any]
) -> np.ndarray:
    """
    Denormalize action data from TeleAvatar's normalized space back to original scale.
    
    Args:
        normalized_action: Normalized action data in range [-1, 1]
        norm_stats: Normalization statistics containing 'q01', 'q99', and optional 'mask'
    
    Returns:
        np.ndarray: Denormalized action data in original scale
    """
    mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    
    # x = (normalized + 1) * (high - low) / 2 + low
    denormalized_action = np.where(
        mask,
        (normalized_action + 1) * (action_high - action_low) / 2 + action_low,
        normalized_action,
    )
    
    return denormalized_action

