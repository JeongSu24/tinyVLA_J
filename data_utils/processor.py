import copy
from dataclasses import dataclass
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import sys
import torch

import transformers

from llava_pythia.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from torch.utils.data import DataLoader, Dataset, Subset
from llava_pythia.train.llava_pythia_trainer import LLaVAPythiaTrainer

from llava_pythia import conversation as conversation_lib
from llava_pythia.model import *
from llava_pythia.mm_utils import tokenizer_image_token
from transformers import CLIPVisionConfig, SiglipVisionConfig, CLIPImageProcessor, SiglipImageProcessor
from PIL import Image
import numpy as np
import os

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args,
) -> Dict:
    """멀티모달 입력에서 이미지 토큰 래핑/치환."""
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v0(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """대화 템플릿을 적용하고 토크나이즈 + Loss 마스킹."""
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if 'phi' in tokenizer.name_or_path.lower():
            total_len += 1
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """간단 포맷의 멀티모달 대화 전처리."""
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """대화 전처리 엔트리포인트 (버전/구분자 스타일에 따라 분기)."""
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    elif conversation_lib.default_conversation.version.startswith("v0"):
        return preprocess_v0(sources, tokenizer, has_image=has_image)
    else:
        raise ValueError(f"Invalid version: {conversation_lib.default_conversation.version}")


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_type: str,
                 data_ratio: int,
                 concat: str,
                 data_args,
                 use_state: bool = False):  # use_state 기본 False
        self.use_state = use_state

        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.concat = concat

        image_file = self.list_data_dict[0]['image']
        image_folder = self.data_args.image_folder
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image_r = Image.open(os.path.join(image_folder, image_file.replace('left_cap2', 'right_cap2'))).convert('RGB')
        print(
            f"{data_type}:Formatting inputs...Skip in lazy mode:{len(list_data_dict)} Size of left single image:{np.array(image).shape};Size of right single image:{np.array(image_r).shape}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def parse_image(self, i, image_file):
        if isinstance(image_file, str):
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        elif isinstance(image_file, torch.Tensor):
            image = Image.fromarray(image_file.numpy())
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # 상태/액션은 필요할 때만 안전하게 로드
        action = sources.get('action', None)
        state = sources.get('state', None) if self.use_state else None

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image = self.parse_image(i, image_file)
            if self.concat != 'single':
                assert 'left_cap2' in self.list_data_dict[i]['image'], f"Wrong data, no left_cap2 in path"
                image_file_right = self.list_data_dict[i]['image'].replace('left_cap2', 'right_cap2')
                image_right = self.parse_image(i, image_file_right)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            )

        # 이미지 포함
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            if self.concat not in ['single', 'direct_cat']:
                data_dict['image_r'] = image_right
        elif self.data_args.is_multimodal:
            try:
                crop_size = self.data_args.image_processor.crop_size
            except:
                crop_size = self.data_args.image_processor.size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        # 로봇 관련 정보 포함
        if action is not None:
            data_dict['action'] = action
        if self.use_state and state is not None:
            data_dict['state'] = state

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """SFT용 배치 콜레이터. use_state에 따라 states를 수집/대체."""
    tokenizer: transformers.PreTrainedTokenizer
    use_state: bool = False
    state_dim: int = 7  # qpos dim 기본 7

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 텍스트 패딩
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        # 액션 수집
        if not isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.tensor(np.array([instance['action'] for instance in instances]))
        else:
            actions = torch.stack([instance['action'] for instance in instances])

        # 상태 수집 (use_state=False면 제로 텐서로 대체)
        if self.use_state and ('state' in instances[0]):
            if not isinstance(instances[0]['state'], torch.Tensor):
                states = torch.tensor(np.array([instance['state'][0:] for instance in instances]))
            else:
                states = torch.stack([instance['state'][0:] for instance in instances])
        else:
            B = len(instances)
            states = torch.zeros((B, self.state_dim), dtype=actions.dtype, device=actions.device)

        # is_pad (있을 때만)
        is_pad_all = None
        if 'is_pad' in instances[0]:
            is_pad_all = torch.stack([instance['is_pad'] for instance in instances])

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            actions=actions,
            states=states,
        )

        if is_pad_all is not None:
            batch['is_pad'] = is_pad_all

        # 이미지들
        if 'image' in instances[0]:
            images = [instance['image'].squeeze() for instance in instances]
            if 'image_r' in instances[0].keys():
                images_right = [instance['image_r'].squeeze() for instance in instances]
            if 'image_top' in instances[0].keys():
                images_top = [instance['image_top'].squeeze() for instance in instances]

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                if 'image_r' in instances[0].keys():
                    batch['images_r'] = torch.stack(images_right)
                if 'image_top' in instances[0].keys():
                    batch['images_top'] = torch.stack(images_top)
            else:
                batch['images'] = images

        # NaN 방지 (텐서인 것만)
        for key in ['actions', 'images', 'images_r', 'images_top']:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = torch.nan_to_num(batch[key])

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, concat="None") -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_eval_split = 0.9
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_ratio=train_eval_split,
                                          data_type='train',
                                          data_path=data_args.data_path,
                                          data_args=data_args,
                                          concat=concat, use_state=data_args.use_state)
    assert 'train' in data_args.data_path or 'eval' in data_args.data_path, "Please use train eval split data!!!!!"
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                         data_ratio=train_eval_split,
                                         data_type='eval',
                                         data_path=data_args.data_path.replace('train', 'eval'),
                                         data_args=data_args,
                                         concat=concat, use_state=data_args.use_state)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        use_state=data_args.use_state,   # 플래그 전달
        state_dim=7                      # 필요 시 설정에서 가져오도록 변경 가능
    )

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
