import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
from data_utils.processor import preprocess, preprocess_multimodal
import copy
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    """
    에피소드형 로봇 데이터셋.
    - 멀티뷰 이미지(예: left/right/top), 액션 시퀀스, (옵션) 로봇 상태(qpos)를 반환
    - HDF5에서 로드하고, chunk 길이에 맞춰 패딩/정규화 수행
    """
    def __init__(self,
                 dataset_path_list,
                 camera_names,
                 norm_stats,
                 episode_ids,
                 episode_len,
                 chunk_size,
                 policy_class,
                 llava_pythia_process=None,
                 imsize=480,
                 use_state=False):
        super(EpisodicDataset).__init__()
        # ------------ 기본 설정 ------------
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.llava_pythia_process = llava_pythia_process
        self.imsize = imsize
        self.use_state = use_state
        print(f"✅ [Dataset] use_state = {self.use_state}")

        # (참고) 프리트레인 데이터 사이즈가 320×180인 경우 알림
        if self.imsize == 320:
            print("########################Current Image Size is [180,320]###################################")

        # diffusion 계열은 간단한 증강 켜기
        self.augment_images = True if 'diffusion' in self.policy_class else False
        self.transformations = None

        # ------------ 3뷰 감지(안전) ------------
        # llava_pythia_process가 있을 때만 한 번 불러서 파이프라인/뷰 수 확인
        self.is_sim = False
        if self.llava_pythia_process is not None:
            try:
                a = self.__getitem__(0)
                if isinstance(a, dict) and ('image_top' in a):
                    print("%"*40)
                    print("There are three views: left, right, top")
            except Exception as _:
                # 초기에 토크나이저/프로세서 준비 전일 수 있으므로 무시
                pass

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        """글로벌 인덱스를 (에피소드ID, 에피소드 내 시작 타임스텝)으로 매핑."""
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # 첫 True 인덱스
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        """
        하나의 샘플을 구성:
          - (K,C,H,W) 멀티뷰 이미지 1프레임 (좌/우/탑)
          - qpos 1스텝(옵션: use_state=True일 때)
          - start_ts 이후 액션 시퀀스 T(패딩 포함)
          - is_pad 마스크
          - raw_lang (언어 지시문)
        """
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]

        with h5py.File(dataset_path, 'r') as root:
            try:  # 일부 구버전 데이터는 sim 속성 없음
                is_sim = root.attrs['sim']
            except Exception:
                is_sim = False
            compressed = root.attrs.get('compress', False)

            raw_lang = root['language_raw'][0].decode('utf-8')

            action = root['/action'][()]
            original_action_shape = action.shape
            episode_len = original_action_shape[0]

            # 관측(1프레임)
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]  # (현재 미사용)
            image_dict = {}
            for cam_name in self.camera_names:
                img = root[f'/observations/images/{cam_name}'][start_ts]
                # 주의: 기존 코드의 (320,180) 리사이즈 하드코딩 유지
                if self.imsize != img.shape[1]:
                    img = cv2.resize(img, (320, 180))
                image_dict[cam_name] = img

            if compressed:
                # JPEG처럼 압축되어 저장된 경우 복원
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

            # 액션 시퀀스 (시뮬/리얼 정렬 차이 보정)
            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):]  # 한 스텝 당겨 정렬
                action_len = episode_len - max(0, start_ts - 1)

        # 패딩 및 길이 잘라내기
        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len, dtype=np.float32)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]

        # 멀티뷰 스택 (K, H, W, C)
        all_cam_images = np.stack([image_dict[cam] for cam in self.camera_names], axis=0)

        # 텐서 변환
        image_data = torch.from_numpy(all_cam_images)              # (K, H, W, C) uint8
        qpos_data  = torch.from_numpy(qpos).float()                # (7,)
        action_data = torch.from_numpy(padded_action).float()      # (T, 10)
        is_pad = torch.from_numpy(is_pad).bool()                   # (T,)

        # (K,H,W,C) → (K,C,H,W) → float32 [0,1]
        image_data = torch.einsum('k h w c -> k c h w', image_data).float() / 255.0

        # 증강 파이프라인 초기화
        if self.transformations is None:
            print('Initializing transformations')
            original_size = image_data.shape[2:]
            ratio = 0.95
            self.transformations = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize(original_size, antialias=True),
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            ]

        # (옵션) 이미지 증강
        if self.augment_images:
            for transform in self.transformations:
                image_data = transform(image_data)


        # 액션 정규화
        if 'diffusion' in self.policy_class:
            # [-1, 1]로 스케일
            action_data = ((action_data - self.norm_stats["action_min"]) /
                           (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
        else:
            # 표준화
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # qpos 표준화
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # ACT 정책이면 기존 튜플 반환
        if self.policy_class == 'ACT':
            return image_data, qpos_data, action_data, is_pad

        # 샘플 구성 (processor로 넘길 입력 형태)
        sample = {
            'image': image_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang
        }
        # use_state 정책에 따라 state 동봉
        if self.use_state:
            sample['state'] = qpos_data

        # processor 쪽 플래그가 있으면 우선 사용, 없으면 데이터셋 플래그 사용
        if hasattr(self.llava_pythia_process, 'data_args') and hasattr(self.llava_pythia_process.data_args, 'use_state'):
            sample['use_state'] = self.llava_pythia_process.data_args.use_state
        else:
            sample['use_state'] = self.use_state

        return self.llava_pythia_process.forward_process(sample)


class LlavaPythiaProcess:
    """
    (백업/참고용) utils.py 내부 프로세서.
    일반적으로 외부에서 주입되는 프로세서를 사용.
    """
    def __init__(self, data_args=None, tokenizer=None, language=None):
        super().__init__()
        self.data_args = data_args
        self.processor = self.data_args.image_processor
        self.tokenizer = tokenizer

    def parse_image(self, image_file):
        """
        (K,C,H,W) 텐서를 받아 배치 전처리(B, C, H, W)로 변환
        - aspect_ratio=='pad'면 정사각 패딩 후 프로세서 적용
        """
        image = image_file
        if isinstance(image, torch.Tensor):
            # (K,C,H,W) → (K,H,W,C) → np
            image = image.permute(0, 2, 3, 1).numpy()

        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square_batch_numpy(pil_imgs, background_color):
                b, h, w, c = pil_imgs.shape
                max_dim = max(h, w)
                expanded = np.full((b, max_dim, max_dim, c), background_color, dtype=np.float32)
                if h == w:
                    expanded[:, :h, :w] = pil_imgs
                elif h > w:
                    off = (max_dim - w) // 2
                    expanded[:, :h, off:off + w] = pil_imgs
                else:
                    off = (max_dim - h) // 2
                    expanded[:, off:off + h, :w] = pil_imgs
                return expanded

            image = expand2square_batch_numpy(image, tuple(x for x in self.processor.image_mean))
            image = self.processor.preprocess(
                image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
            )['pixel_values']   # (B, C, H, W)
        else:
            image = self.processor.preprocess(
                image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
            )['pixel_values']
        return image

    def forward_process(self, sample):
        """
        데이터 샘플을 LLaVA-Pythia 입력 포맷으로 변환.
        - 텍스트 토크나이즈 + 이미지 전처리 + (옵션) 상태/액션/패딩 마스크 포함
        """
        sources = self.datastruct_droid2llava(sample)
        image = self.parse_image(sample['image'])

        if not isinstance(sources, list):
            sources = [sources]
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        tokenized = preprocess(sources, self.tokenizer, has_image=True)
        data_dict = dict(input_ids=tokenized["input_ids"][0],
                         labels=tokenized["labels"][0])

        # 멀티뷰 분리
        images_all = torch.chunk(image, image.shape[0], dim=0)
        data_dict['image'] = images_all[0]
        data_dict['image_r'] = images_all[1]
        if image.shape[0] == 3:
            data_dict['image_top'] = images_all[2]

        # use_state 정책을 존중해서 state 첨부
        if sample.get('use_state', True) and ('state' in sample):
            data_dict['state'] = sample['state']
        data_dict['action'] = sample['action']
        data_dict['is_pad'] = sample['is_pad']
        return data_dict

    def datastruct_droid2llava(self, sample):
        """
        DROID식 샘플을 LLaVA 대화 포맷으로 매핑.
        - 첫 메시지에 <image> 토큰과 자연어 지시문 포함
        """
        sources = {
            'id': "",
            'image': None,
            'state': [],
            'action': [],
            "conversations": [
                {"from": "human", "value": "<image>\n"},
                {"from": "gpt",   "value": " "}
            ]
        }
        if 'action' in sample:
            sources['action'] = sample['action']
        if 'state' in sample:
            sources['state'] = sample['state']
        sources["conversations"][0]["value"] += sample['raw_lang']
        return sources


def get_norm_stats(dataset_path_list):
    """
    qpos/액션 정규화 통계와 각 에피소드 길이 계산.
    """
    all_qpos_data, all_action_data, all_episode_len = [], [], []
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e); quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # 액션 표준화 통계
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std  = all_action_data.std(dim=[0]).float()
    action_std  = torch.clip(action_std, 1e-2, np.inf)

    # qpos 표준화 통계
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std  = all_qpos_data.std(dim=[0]).float()
    qpos_std  = torch.clip(qpos_std, 1e-2, np.inf)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 1e-4
    stats = {
        "action_mean": action_mean.numpy(),
        "action_std":  action_std.numpy(),
        "action_min": (action_min.numpy() - eps),
        "action_max": (action_max.numpy() + eps),
        "qpos_mean":   qpos_mean.numpy(),
        "qpos_std":    qpos_std.numpy(),
        "example_qpos": qpos,
    }
    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    """디렉토리에서 .hdf5 파일 목록을 재귀적으로 수집."""
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: 
                continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    """에피소드 길이/가중치에 따라 step 인덱스를 샘플링하는 제네레이터."""
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names,
              batch_size_train, batch_size_val, chunk_size, config,
              skip_mirrored_data=False, policy_class=None, stats_dir_l=None,
              sample_weights=None, train_ratio=0.99, return_dataset=False,
              llava_pythia_process=None, use_state=False):
    """
    학습/검증 데이터셋 로딩 및 정규화 통계 계산.
    - utils.py 버전도 processor/dataset 변경에 맞춰 use_state를 전달하도록 수정.
    """
    if isinstance(dataset_dir_l, str):
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(d) for d in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # 첫 디렉토리 기준 train/val split
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0   = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(num) + num_episodes_cumsum[idx] for idx, num in enumerate(num_episodes_l[1:])
    ]
    val_episode_ids_l   = [val_episode_ids_0]

    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids   = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # 길이/정규화 통계
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l   = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len   = flatten_list(train_episode_len_l)
    val_episode_len     = flatten_list(val_episode_len_l)

    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif isinstance(stats_dir_l, str):
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')
    print(f'train_episode_len_l: {train_episode_len_l}')

    # use_state를 명시적으로 전달
    train_dataset = EpisodicDataset(
        dataset_path_list, camera_names, norm_stats,
        train_episode_ids, train_episode_len, chunk_size, policy_class,
        llava_pythia_process=llava_pythia_process,
        imsize=config['training_args'].pretrain_image_size,
        use_state=use_state
    )
    val_dataset = EpisodicDataset(
        dataset_path_list, camera_names, norm_stats,
        val_episode_ids, val_episode_len, chunk_size, policy_class,
        llava_pythia_process=llava_pythia_process,
        imsize=config['training_args'].pretrain_image_size,
        use_state=use_state
    )

    sampler_params = {
        'train': {"batch_size": batch_size_train, 'episode_len_l': train_episode_len_l, 'sample_weights': sample_weights},
        'eval':  {"batch_size": batch_size_val,   'episode_len_l': val_episode_len_l,   'sample_weights': None}
    }

    if return_dataset:
        return train_dataset, val_dataset, norm_stats, sampler_params


# ===== (옵션) 모바일 베이스 로봇 유틸 =====

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0  # 0.19 등 보정 상수
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    return smooth_base_action(base_action)

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    return np.array([linear_vel, angular_vel])

# ===== 헬퍼 =====

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    return {k: v.detach() for k, v in d.items()}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
