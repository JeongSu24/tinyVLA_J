import itertools
import os
import os.path as osp
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from tqdm import tqdm

import h5py

import torch
import collections
import tensorflow_datasets as tfds
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import time
import argparse

# 이미지 폴더 경로에서 이미지 파일들을 불러와 numpy 배열로 변환
# remove_index_list에 포함된 인덱스는 제외
def get_image_list_np(img_rgb_dir_path, remove_index_list):
    cur_camera_rgb_list = []
    img_name_list = os.listdir(img_rgb_dir_path)  # 이미지 파일 이름 목록 가져오기
    img_name_list = sorted(img_name_list)  # 정렬

    for idx, img_name in enumerate(img_name_list):
        if idx in remove_index_list:
            continue

        img_path = os.path.join(img_rgb_dir_path, img_name)

        img_frame = Image.open(img_path).convert('RGB')  # 이미지 열고 RGB로 변환
        img_np = np.array(img_frame)  # numpy 배열로 변환
        cur_camera_rgb_list.append(img_np)

    cur_camera_rgb_np = np.array(cur_camera_rgb_list)
    print('+++++++++++++++')
    print(f"img_rgb_dir_path: {img_rgb_dir_path}")
    print(f'cur_camera_rgb_np size: {cur_camera_rgb_np.shape}')

    return cur_camera_rgb_np

# 예측 행동을 x, y, z 별로 시각화하여 저장하는 함수
def plot_smooth_action(traj_act_xyz_np, fig_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    figure_name = ["x", "y", "z"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(range(traj_act_xyz_np.shape[0]), traj_act_xyz_np[:, i], label='cur_action')
        plt.title(figure_name[i])
        plt.legend()
    plt.suptitle(f"Differences between predicted and target actions_traj")
    plt.tight_layout()

    work_dir = '/home/jz08/wk/datasets/real_franka/ori_datasets'
    figure_dir_path = os.path.join(work_dir, f"smooth_action_results")
    os.makedirs(figure_dir_path, exist_ok=True)
    figure_path = os.path.join(figure_dir_path, f"{fig_name}.png")
    plt.savefig(figure_path)
    plt.clf()

# h5py Group 내부 구조를 프린트하는 재귀 함수
def print_h5_structure(group, indent=0):
    for name in group:
        item = group[name]
        print(" " * indent + f"name: {name}")
        if isinstance(item, h5py.Group):
            print(" " * indent + f"Group: {name}")
            print_h5_structure(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            print(" " * indent + f"Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
        else:
            print(" " * indent + f"Unknown item: {name}")

# 딕셔너리 구조를 출력하는 재귀 함수
def print_dict_structure(cur_dict, indent=0):
    for name in cur_dict.keys():
        item = cur_dict[name]
        print(" " * indent + f"name: {name}")
        if isinstance(item, dict):
            print(" " * indent + f"Dict: {name}")
            print_dict_structure(item, indent + 2)
        elif isinstance(item, np.ndarray):
            print(" " * indent + f"Array: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
        else:
            print(" " * indent + f"Unknown item: {name}")

# torch Tensor를 numpy로 변환하는 함수 (중첩 구조 포함)
def to_numpy(x):
    def f(tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()

    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: f,
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        }
    )

# 중첩 구조에서 각 타입별로 지정된 함수 적용 (재귀)
def recursive_dict_list_tuple_apply(x, type_func_dict):
    assert (list not in type_func_dict)
    assert (tuple not in type_func_dict)
    assert (dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            return x

# 회전 행렬을 6D 회전 표현으로 변환
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

# 오일러 각도를 6D 회전 표현으로 변환
def euler_angles_to_rot_6d(euler_angles, convention="XYZ"):
    rot_mat = euler_angles_to_matrix(euler_angles, convention="XYZ")
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d

# 하나의 축 기준으로 회전 행렬 생성
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

# 오일러 각도를 회전 행렬로 변환
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

# h5py Group 구조를 numpy dict로 변환
def convert_h5py2np_dict(group, state_np_dict, indent=0):
    for name in group:
        item = group[name]
        print(" " * indent + f"name: {name}")
        if isinstance(item, h5py.Group):
            state_np_dict[name] = dict()
            sub_np_dict = state_np_dict[name]
            print(" " * indent + f"Group: {name}")
            convert_h5py2np_dict(item, sub_np_dict, indent + 2)
        elif isinstance(item, h5py.Dataset):
            state_np_dict[name] = item[...]
            tmp = state_np_dict[name]
            print(" " * indent + f"Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
            print(" " * indent + f"Array: {name} (Shape: {tmp.shape}, Dtype: {tmp.dtype})")
        else:
            state_np_dict[name] = item
            print(" " * indent + f"Unknown item: {name}")


def print_name(name):
    print(name)  # 이름 출력 함수

# HDF5 파일 생성 함수
# 주어진 관측(observation) 및 행동(action) 데이터를 기반으로 HDF5 파일을 생성하여 저장
# language instruction도 함께 저장됨
def generate_h5(obs_replay, action_replay, cfg, total_traj_cnt, act_root_dir_path, edit_flag):
    data_dict = {
        '/observations/qpos': obs_replay['qpos'],
        '/observations/qvel': obs_replay['qvel'],
        '/action': action_replay,
        'is_edited': np.array(edit_flag)
    }
    for cam_name in cfg['camera_names']:
        data_dict[f'/observations/images/{cam_name}'] = obs_replay['images'][cam_name]

    max_timesteps = len(data_dict['/observations/qpos'])
    print(f'max_timesteps: {max_timesteps}')

    dataset_path = os.path.join(act_root_dir_path, f'episode_{total_traj_cnt}')
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True  # 시뮬레이션 여부
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in cfg['camera_names']:
            _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3),
                                     dtype='uint8', chunks=(1, cfg['cam_height'], cfg['cam_width'], 3))
        qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
        qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
        action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
        is_edited = root.create_dataset('is_edited', (1))
        raw_lang = cfg['lang_intrs']
        root.create_dataset("language_raw", data=[raw_lang])  # 언어 지시문 저장

        print(f'==== generate h5 ======')
        for name, array in data_dict.items():
            print(f"name: {name}")
            print(f"array: {array.shape}")
            root[name][...] = array

# gif 저장용 임시 함수 (tkinter용 입력 대기창 구현 가능)
user_input = None

def show_gif(images):
    path = os.path.join('./temp.gif')
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / 15), loop=0)

# 환경/프레임워크 및 데이터셋 설정
cfg = {
    "task_name": "droid_1dot7t_lang",
    "camera_names": ["left", "right"],  # 관측 카메라 이름들
    "dataset_dir": "/home/jz08/wk/datasets/real_franka/act_datasets",
    "cam_height": 180,
    "cam_width": 320,
    "state_dim": 7,
    "action_dim": 10,
    "lang_intrs": 'close the lid of the box'  # 기본 언어 지시문
}

raw_lang = cfg['lang_intrs']
print('raw_lang: {raw_lang}')
task_name = cfg['task_name']
parser = argparse.ArgumentParser()

# TFDS 소스 데이터 루트 및 이름 정의
parser.add_argument("--src_root", default= '/media/rl/PSSD-6')
parser.add_argument("--name", default="droid")
args = parser.parse_args()

# 변환된 HDF5 저장 경로 설정
act_target_root = os.path.join(args.src_root, "droid_with_lang")
os.makedirs(act_target_root, exist_ok=True)

# 부드러운 행동 설정
smooth_action = False
smooth_order = 0
smooth_window_size = 0

framework = 'droid'
act_pos_thres = 0.001

# 데이터셋 출력 디렉토리 설정
dataset_subdir = f'{task_name}_succ_t0001_s-{smooth_window_size}-{smooth_order}'
act_root_dir_path = os.path.join(act_target_root, dataset_subdir)
os.makedirs(act_root_dir_path, exist_ok=True)

IMAGE_NAME_TO_CAM_KEY_MAPPING = dict()

# 전체 에피소드 카운터 초기화
succ_traj_count = 0
fail_traj_count = 0
total_traj_cnt = 0

# 통계 저장용 변수 초기화
max_action_np = None
min_action_np = None
data_normalize_stats = dict()
all_traj_state_total_np_dict = dict()

# TFDS 데이터셋 로딩 (train split)
ds = tfds.load(args.name, data_dir=args.src_root, split="train")
for episode in tqdm(ds):  # 에피소드별 반복

    if os.path.exists(os.path.join(act_root_dir_path, f'episode_{total_traj_cnt}.hdf5')):
        total_traj_cnt += 1
        continue

    # 에피소드 내 변수 초기화
    state_total_np_dict = dict()
    cur_actions = []
    cur_obs_image = {'1': [], '2': []}  # 좌/우 이미지
    cur_obs_gripper_pos = []
    cur_obs_joint_state = []
    cur_obs_cartesian_position = []
    raw_lang = ""
    cur_actions_dict = {}
    edit_flag = 0

    # 에피소드 내부 프레임 순회
    for idx, step in enumerate(episode['steps']):
        if idx == 0:
            cur_actions_dict = {k: [] for k in step['action_dict'].keys()}

        # 언어 지시문이 비어있으면 제거 대상 표시
        if len(step['language_instruction'].numpy().decode('utf-8')) < 4 or \
           len(step['language_instruction_2'].numpy().decode('utf-8')) < 4 or \
           len(step['language_instruction_3'].numpy().decode('utf-8')) < 4:
            print("No langauge instruction in this episode....")
            edit_flag = 1

        # 각 관측값 저장
        cur_actions.append(step['action'].numpy()[:-1])
        cur_obs_image['1'].append(step['observation']['exterior_image_1_left'].numpy())
        cur_obs_image['2'].append(step['observation']['exterior_image_2_left'].numpy())
        cur_obs_gripper_pos.append(step['observation']['gripper_position'].numpy())
        cur_obs_joint_state.append(step['observation']['joint_position'].numpy())
        raw_lang = step['language_instruction'].numpy().decode('utf-8')
        cur_obs_cartesian_position.append(step['observation']['cartesian_position'].numpy())

        for k, v in cur_actions_dict.items():
            cur_actions_dict[k].append(step['action_dict'][k].numpy())

    # 언어 지시문 수동 입력 받기 (비었을 경우)
    if idx == 0 or edit_flag == 1:
        all_images_np = np.concatenate((np.array(cur_obs_image['1']), np.array(cur_obs_image['2'])), axis=2)
        all_images = [Image.fromarray(each) for each in all_images_np]
        show_gif(all_images)
        raw_lang = input("please write a language instruction:")
        print(f"Your input: {raw_lang}")

    # 관측 및 행동 데이터 정리 및 변환
    action_dict_group = {k: np.array(v) for k, v in cur_actions_dict.items()}
    cur_obs_image = {k: np.array(v) for k, v in cur_obs_image.items()}
    cur_obs_cartesian_position = torch.from_numpy(np.array(cur_obs_cartesian_position))
    state_total_np_dict['act_gripper_position'] = action_dict_group['gripper_position']
    state_total_np_dict['obs_cartesian_position'] = cur_obs_cartesian_position
    state_total_np_dict['obs_gripper_position'] = torch.from_numpy(np.array(cur_obs_gripper_pos))
    state_total_np_dict['obs_joint_positions'] = torch.from_numpy(np.array(cur_obs_joint_state))

    # 절대/상대 위치 및 회전을 euler -> 6D 변환
    for in_ac_key in ["cartesian_position", "cartesian_velocity"]:
        in_action = action_dict_group[in_ac_key][:]
        in_pos = in_action[:, :3].astype(np.float64)
        in_rot = in_action[:, 3:6].astype(np.float64)
        rot_ = torch.from_numpy(in_rot)
        rot_6d = euler_angles_to_rot_6d(rot_, convention="XYZ").numpy().astype(np.float64)

        prefix = "act_abs_" if in_ac_key == "cartesian_position" else "act_rel_"
        this_action_dict = {
            prefix + 'pos': in_pos,
            prefix + 'rot_euler': in_rot,
            prefix + 'rot_6d': rot_6d,
        }
        for key, data in this_action_dict.items():
            print(f'action key: {key}, value shape: {data.shape}')
            state_total_np_dict[key] = data

    traj_len = min(len(cur_obs_image['1']), len(cur_obs_image['2']))
    print('********** final process ***********')
    print(f'print state_total_np_dict:')
    for key, data in state_total_np_dict.items():
        print(f"key: {key}")
        print(f"data shape: {data.shape}")
    print(f'traj_len: {traj_len}')

    # 최종 행동 벡터 구성 (xyz + rot6d + gripper)
    traj_xyz = state_total_np_dict['act_abs_pos']
    traj_rot = state_total_np_dict['act_abs_rot_6d']
    traj_gripper = state_total_np_dict['act_gripper_position']
    traj_actions = np.concatenate((traj_xyz, traj_rot, traj_gripper), axis=-1)[:traj_len]
    print(f"traj_actions shape: {traj_actions.shape}")

    # 관측값(qpos, qvel)
    traj_qpos = np.concatenate((state_total_np_dict['obs_cartesian_position'], state_total_np_dict['obs_gripper_position']), axis=-1)[:traj_len]
    traj_qvel = np.zeros_like(traj_qpos)
    print(f"traj_qpos shape: {traj_qpos.shape}")
    print(f"traj_qvel shape: {traj_qvel.shape}")

    # 이미지 프레임
    left_imgs = cur_obs_image['1'][:traj_len]
    right_imgs = cur_obs_image['2'][:traj_len]
    print(f"left_imgs shape: {left_imgs.shape}")
    print(f"right_imgs shape: {right_imgs.shape}")

    # 관측 딕셔너리 구성
    obs_replay = {
        'qpos': traj_qpos,
        'qvel': traj_qvel,
        'images': {'left': left_imgs, 'right': right_imgs}
    }
    cfg['lang_intrs'] = raw_lang
    print(raw_lang)

    # HDF5 파일 생성 및 저장
    generate_h5(obs_replay, traj_actions, cfg, total_traj_cnt, act_root_dir_path, edit_flag)
    total_traj_cnt += 1

