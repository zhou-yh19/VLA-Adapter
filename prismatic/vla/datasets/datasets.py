"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    NUM_STAGES,
    PROPRIO_DIM,
    STAGE_PLACEHOLDER_ID,
    STOP_INDEX,
    NUM_TOKENS,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def extract_last_number(line: str) -> Optional[int]:
    """
    从单行提示词提取最后一个数字。
    - current task: Task4 → 4
    - current task: None 且 completed tasks: task1, task2, task3 → 3
    """
    m = re.search(r"current task:\s*(task(\d+)|None)\s*\.?\s*$", line.strip())
    if not m:
        return None
    if m.group(1) != "None":
        return int(m.group(2))
    comp = re.search(r"completed tasks:\s*(.*?)\.\s*current task:", line)
    if not comp or comp.group(1).strip() == "None":
        return 0
    return len(re.findall(r"task\d+", comp.group(1)))


def _parse_task_states(lang: str) -> Tuple[list, Optional[int]]:
    """
    从 language_instruction 解析：已完成任务列表、当前任务编号。
    - completed tasks: task1, task2 -> [1, 2]；completed tasks: None -> []
    - current task: task3 -> 3；current task: None -> None
    """
    completed = []
    comp_match = re.search(
        r"completed\s+tasks?:\s*(.*?)(?:\.\s*current\s+task:|$)",
        lang,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if comp_match:
        comp_text = comp_match.group(1).strip().lower()
        if comp_text and comp_text != "none":
            completed = [int(m.group(1)) for m in re.finditer(r"task(\d+)", comp_text)]
    current = None
    curr_match = re.search(r"current\s+task:\s*(task(\d+)|None)\s*\.?\s*$", lang.strip(), flags=re.IGNORECASE)
    if curr_match and curr_match.group(1).lower() != "none":
        current = int(curr_match.group(2))
    return completed, current


def add_task_description_suffix(lang: str) -> str:
    """
    将四个任务（task1~task4）标注成三种状态后返回，用作 prompt 的 language 部分：
    - completed tasks 中的任务 -> done
    - current task -> active
    - 其余任务 -> waiting
    若存在 "completed tasks" / "current task" 前的描述文本则保留为前缀。
    """
    completed, current = _parse_task_states(lang)
    # 提取前缀："completed tasks" 之前的内容（若有）
    prefix_match = re.search(r"^(.+?)\s*completed\s+tasks?", lang, flags=re.IGNORECASE | re.DOTALL)
    prefix = prefix_match.group(1).strip() if prefix_match and prefix_match.group(1).strip() else ""

    states = []
    for i in range(1, 5):
        if i in completed:
            states.append(f"task{i}:done")
        elif i == current:
            states.append(f"task{i}:active")
        else:
            states.append(f"task{i}:waiting")
    return (prefix + ",".join(states)).strip()


def get_task_description(lang: str) -> Tuple[str, str]:
    """Split language_instruction into (high_level_task, low_level_task).

    Returns:
        high_level_task: prefix description before "completed tasks" (e.g. "organize the desk.")
        low_level_task:  comma-separated stage states (e.g. "task1:done,task2:active,task3:waiting,task4:waiting")
    """
    completed, current = _parse_task_states(lang)
    prefix_match = re.search(r"^(.+?)\s*completed\s+tasks?", lang, flags=re.IGNORECASE | re.DOTALL)
    prefix = prefix_match.group(1).strip() if prefix_match and prefix_match.group(1).strip() else ""

    states = []
    for i in range(1, 5):
        if i in completed:
            states.append(f"task{i}:done")
        elif i == current:
            states.append(f"task{i}:active")
        else:
            states.append(f"task{i}:waiting")
    return prefix, ",".join(states)


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False
    use_high_level_task_only: bool = False


    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # 从 language_instruction 解析 stage：仅当解析结果为 1–4 时添加 stage 特征（非 RLDS 必须）
        # CrossEntropy 要求类别下标为 0~num_classes-1，故存 0-indexed（0,1,2,3），不能存 1~4
        stage_raw = extract_last_number(lang)
        stage_class_index = (stage_raw - 1) if (stage_raw is not None and 1 <= stage_raw <= 4) else None
        
        if self.use_high_level_task_only:
            if stage_class_index is not None:
                high_level_task, _ = get_task_description(lang)
                lang_for_prompt = high_level_task
            else:
                lang_for_prompt = lang
        else:
            # 做 stage 预测时：prompt 中不包含 "Current task"，只保留已完成任务，迫使模型依赖视觉+语言预测当前任务
            lang_for_prompt = add_task_description_suffix(lang) if stage_class_index is not None else lang

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]

        if self.use_minivlm:
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn("openvla")
            # Get action chunk string
            future_actions_string = self.action_tokenizer(future_actions,self.use_minivlm)
            current_action_string = self.action_tokenizer(current_action,self.use_minivlm)

            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
            action_chunk_len = len(flattened_action_chunk_string) 

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang_for_prompt}?"},
                {"from": "gpt", "value": ''},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

            if len(input_ids) >= 3:
                del input_ids[-3] 
                del input_ids[-2] 
                del input_ids[-1] 

            if NUM_TOKENS<len(flattened_action_chunk_string):
                input_ids = input_ids + flattened_action_chunk_string[:NUM_TOKENS]
            else:
                remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
                extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
                
                input_ids = input_ids + flattened_action_chunk_string + extended_array
            labels = list(input_ids)
            action_chunk_len = NUM_TOKENS

        else:
            future_actions_string = ''.join(self.action_tokenizer(future_actions, use_minivlm=False))

            # Get action chunk string
            current_action_string = self.action_tokenizer(current_action, use_minivlm=False)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang_for_prompt}?"},
                {"from": "gpt", "value": action_chunk_string[0]},
            ]
            # remove action token
            # conversation = [
            #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
            #     {"from": "gpt", "value": ""},
            # ]
            action_chunk_len = 1


            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
        )
        if stage_class_index is not None:
            return_dict["stage"] = torch.tensor(stage_class_index, dtype=torch.long)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


# Action placeholder token ID used as position marker for _process_action_masks.
# Any value > ACTION_TOKEN_BEGIN_IDX works; the actual embedding is replaced by action_queries in forward().
_ACTION_PLACEHOLDER_ID = ACTION_TOKEN_BEGIN_IDX + 1


@dataclass
class RLDSBatchTransform4VLAAdapterStage:
    """Batch transform for VLA-Adapter Stage training with L1 regression.

    Produces input_ids with the layout:
        [high_level_prompt] [stage_placeholder×8] [low_level_prompt] [action_placeholder×64]
    After model forward (vision insertion + embedding replacement) the LLM sees:
        [BOS] [patches] [high_level_prompt] [stage_queries×8] [low_level_prompt] [action_queries×64]
    """
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # --- Parse stage label (0-indexed: 0,1,2,3) ---
        stage_raw = extract_last_number(lang)
        stage_class_index = (stage_raw - 1) if (stage_raw is not None and 1 <= stage_raw <= 4) else None

        # --- Split task description ---
        if stage_class_index is not None:
            high_level_task, low_level_task = get_task_description(lang)
        else:
            high_level_task, low_level_task = lang, ""

        # --- Encode high-level prompt (system + user question + assistant start) ---
        high_level_prompt = (
            f"<|im_start|>system\n"
            f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"What action should the robot take to {high_level_task.lower()}?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        high_ids = self.base_tokenizer(high_level_prompt, add_special_tokens=True).input_ids

        # --- Stage placeholders (replaced by stage_queries in model forward) ---
        stage_ids = [STAGE_PLACEHOLDER_ID] * NUM_STAGES

        # --- Encode low-level prompt (no special tokens to avoid duplicate BOS) ---
        low_level_text = low_level_task.lower()+"<|im_end|>\n"
        low_ids = self.base_tokenizer(low_level_text, add_special_tokens=False).input_ids

        # --- Action placeholders (replaced by action_queries in model forward) ---
        action_ids = [_ACTION_PLACEHOLDER_ID] * NUM_TOKENS

        # --- Assemble: high | stage | low | action ---
        stage_start_idx = len(high_ids)
        input_ids = high_ids + stage_ids + low_ids + action_ids
        labels = list(input_ids)

        # --- Tensorize ---
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # --- Mask labels: only supervise the 64 action placeholders ---
        labels[:-NUM_TOKENS] = IGNORE_INDEX

        # --- Assemble return dict ---
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
            stage_start_idx=stage_start_idx,
        )

        if stage_class_index is not None:
            return_dict["stage"] = torch.tensor(stage_class_index, dtype=torch.long)

        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)

        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            return_dict["proprio"] = rlds_batch["observation"]["proprio"]

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        elif "right_grip_grab_a_stuffed_animal_into_left_box" in self.data_mix:
            load_camera_views = ("primary", "secondary", "left_wrist", "right_wrist")
        elif "build_blocks" in self.data_mix:
            load_camera_views = ("primary", "secondary", "left_wrist", "right_wrist")
        elif "organize_the_desk" in self.data_mix:
            load_camera_views = ("primary", "secondary", "left_wrist", "right_wrist")
        elif "organize_the_desk_stage" in self.data_mix:
            load_camera_views = ("primary", "secondary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
