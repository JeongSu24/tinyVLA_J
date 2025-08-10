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

def get_image_list_np(img_rgb_dir_path, remove_index_list):
    cur_camera_rgb_list = []
    img_name_list = os.listdir(img_rgb_dir_path)
    img_name_list = sorted(img_name_list)

    for idx, img_name in enumerate(img_name_list):
        if idx in remove_index_list:
            continue

        img_path = os.path.join(img_rgb_dir_path, img_name)

        # (w 640, h 480)
        img_frame = Image.open(img_path).convert('RGB')
        # print(f'cur frame {img_frame} size: {img_frame.size}')
        # (480, 640, 3)
        img_np = np.array(img_frame)
        # print(f'cur np {img_path} size: {img_np.shape}')
        cur_camera_rgb_list.append(img_np)

    cur_camera_rgb_np = np.array(cur_camera_rgb_list)
    print('+++++++++++++++')
    print(f"img_rgb_dir_path: {img_rgb_dir_path}")
    print(f'cur_camera_rgb_np size: {cur_camera_rgb_np.shape}')

    return cur_camera_rgb_np


def plot_smooth_action(traj_act_xyz_np, fig_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    figure_name = ["x", "y", "z"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(range(traj_act_xyz_np.shape[0]), traj_act_xyz_np[:, i], label='cur_action')
        # plt.plot(range(traj_act_xyz_np.shape[0]), traj_act_xyz_np[:, i], label='gt_action')
        plt.title(figure_name[i])
        plt.legend()
    plt.suptitle(f"Differences between predicted and target actions_traj")
    plt.tight_layout()

    # print(f"eval_traj_name: {eval_traj_name}")
    # work_dir = '/home/eai/wk/gitlab/real_franka_dataset/serl_data/serl_datasets/'
    # work_dir = '/data/team/wk/datasets/real_franka/ori_datasets'
    work_dir = '/home/jz08/wk/datasets/real_franka/ori_datasets'
    figure_dir_path = os.path.join(work_dir, f"smooth_action_results")
    os.makedirs(figure_dir_path, exist_ok=True)
    figure_path = os.path.join(figure_dir_path, f"{fig_name}.png")
    plt.savefig(figure_path)
    plt.clf()


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


def to_numpy(x):
    """
    Converts all torch tensors in nested dictionary or list or tuple to
    numpy (and leaves existing numpy arrays as-is), and returns
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """

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


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
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
            ## Pretty hacky fix to avoid error when strings get converted to tensors
            ## TODO (surajnair) try and clean this up at some point
            return x
            # raise NotImplementedError(
            #     'Cannot handle data type %s' % str(type(x)))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def euler_angles_to_rot_6d(euler_angles, convention="XYZ"):
    """
    Converts tensor with rot_6d representation to euler representation.
    """
    rot_mat = euler_angles_to_matrix(euler_angles, convention="XYZ")
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

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


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
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
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


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
    print(name)

def generate_h5(obs_replay, action_replay, cfg, total_traj_cnt, act_root_dir_path, edit_flag):
    """
    Generates an HDF5 file to store observation and action data for a given trajectory.

    Args:
        obs_replay (dict): A dictionary containing observation data, including 'qpos', 'qvel', and 'images'.
        action_replay (numpy.ndarray): An array containing action data for the trajectory.
        cfg (dict): Configuration dictionary containing metadata such as camera names, dimensions, and language instructions.
        total_traj_cnt (int): The current trajectory count, used to name the output file.
        act_root_dir_path (str): The root directory path where the HDF5 file will be saved.
        edit_flag (bool): A flag indicating whether the data has been edited.

    The function creates an HDF5 file named 'episode_{total_traj_cnt}.hdf5' in the specified directory.
    It stores the observation data, action data, and metadata such as whether the data was edited and the raw language instructions.
    """
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

    data_dir = act_root_dir_path
    # create data dir if it doesn't exist
    # data_dir = os.path.join(cfg['dataset_dir'], cfg['task_name'])
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)

    dataset_path = os.path.join(data_dir, f'episode_{total_traj_cnt}')
    # save the data, 2GB cache
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in cfg['camera_names']:
            _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                     chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
        qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
        qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
        # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
        action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
        is_edited = root.create_dataset('is_edited', (1))
        # dt = h5py.special_dtype(vlen=str)
        # dt = h5py.string_dtype()
        # lang_intrs = root.create_dataset('lang_intrs', data=cfg['lang_intrs'], dtype=dt)
        # lang_intrs['/lang_intrs'][...] = cfg['lang_intrs']
        raw_lang = cfg['lang_intrs']
        # encoded_lang = cfg['lang_intrs_distilbert']
        root.create_dataset("language_raw", data=[raw_lang])
        # root.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())

        print(f'==== generate h5 ======')
        for name, array in data_dict.items():
            print(f"name: {name}")
            print(f"array: {array.shape}")
            root[name][...] = array

user_input = None
def show_gif(images):
    path = os.path.join('./temp.gif')
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / 15), loop=0)


# ✅ 수정 1: wrist 카메라까지 포함
cfg = {
    "task_name": "droid_1dot7t_lang",
    "camera_names": ["left", "right", "wrist"],  # ✅ wrist 추가
    "dataset_dir": "/home/jz08/wk/datasets/real_franka/act_datasets",
    "cam_height": 180,
    "cam_width": 320,
    "state_dim": 7,
    "action_dim": 7,  # ✅ rot_6d 제거로 인해 7차원으로 간결화
    "lang_intrs": 'close the lid of the box'
}

# H = f["observation/robot_state/cartesian_position"].shape[0]
raw_lang = cfg['lang_intrs']
print('raw_lang: {raw_lang}')
task_name = cfg[
    'task_name']
parser = argparse.ArgumentParser()

# External config file that overwrites default config
parser.add_argument(
"--src_root", default= '/media/rl/PSSD-6')
parser.add_argument(
"--name", default="droid")

# parser.add_argument(
# "--act_target_root", default='/media/rl/HDD/data/data/act/test')
args = parser.parse_args()

act_target_root = os.path.join(args.src_root, "droid_with_lang")
os.makedirs(act_target_root, exist_ok=True)
smooth_action = False  # True #False #True
smooth_order = 0  # 3 #0 #2 #3 #2 #3

smooth_window_size = 0  # 5 #0 #3 #5 #3 #5 #10 # 'traj_length'

framework = 'droid'  # 'serl' #'droid'
# act_pos_thres = 0.05
# for serl framework
# act_pos_thres = 0.05
# for droid framework
act_pos_thres = 0.001

act_root_dir_name = f'{task_name}_succ_t0001_s-{smooth_window_size}-{smooth_order}'
act_root_dir_path = os.path.join(act_target_root, act_root_dir_name)
os.makedirs(act_root_dir_path, exist_ok=True)

IMAGE_NAME_TO_CAM_KEY_MAPPING = dict()

succ_traj_count = 0
fail_traj_count = 0
total_traj_cnt = 0

max_action_np = None
min_action_np = None
data_normalize_stats = dict()
all_traj_state_total_np_dict = dict()

# src_root_dir_list = os.listdir(src_root)
# src_root_dir_list = sorted(src_root_dir_list)

ds = tfds.load(args.name, data_dir=args.src_root, split="train")
f# ✅ 에피소드 처리 시작
for episode in tqdm(ds):
    if os.path.exists(os.path.join(act_root_dir_path, f'episode_{total_traj_cnt}.hdf5')):
        total_traj_cnt += 1
        continue

    state_total_np_dict = dict()
    cur_actions = []
    cur_obs_image = {'1': [], '2': [], 'wrist': []}  # ✅ wrist 추가
    cur_obs_gripper_pos = []
    cur_obs_joint_state = []
    cur_obs_cartesian_position = []
    raw_lang = ""
    cur_actions_dict = {}
    edit_flag = 0

    for idx, step in enumerate(episode['steps']):
        if idx == 0:
            cur_actions_dict = {k: [] for k in step['action_dict'].keys()}

        if len(step['language_instruction'].numpy().decode('utf-8')) < 4 or \
           len(step['language_instruction_2'].numpy().decode('utf-8')) < 4 or \
           len(step['language_instruction_3'].numpy().decode('utf-8')) < 4:
            print("No langauge instruction in this episode....")
            edit_flag = 1

        # ✅ 행동 벡터 그대로 사용 (7차원)
        cur_actions.append(step['action'].numpy())

        # ✅ 이미지 추가
        cur_obs_image['1'].append(step['observation']['exterior_image_1_left'].numpy())
        cur_obs_image['2'].append(step['observation']['exterior_image_2_left'].numpy())
        cur_obs_image['wrist'].append(step['observation']['wrist_image_left'].numpy())  # ✅ wrist 추가

        cur_obs_gripper_pos.append(step['observation']['gripper_position'].numpy())
        cur_obs_joint_state.append(step['observation']['joint_position'].numpy())
        raw_lang = step['language_instruction'].numpy().decode('utf-8')
        cur_obs_cartesian_position.append(step['observation']['cartesian_position'].numpy())

        for k, v in cur_actions_dict.items():
            cur_actions_dict[k].append(step['action_dict'][k].numpy())

    if idx == 0 or edit_flag == 1:
        all_images_np = np.concatenate((np.array(cur_obs_image['1']), np.array(cur_obs_image['2'])), axis=2)
        all_images = [Image.fromarray(each) for each in all_images_np]
        show_gif(all_images)
        raw_lang = input("please write a language instruction:")
        print(f"Your input: {raw_lang}")

    # ✅ dict → numpy 변환
    cur_obs_image = {k: np.array(v) for k, v in cur_obs_image.items()}
    cur_obs_cartesian_position = torch.from_numpy(np.array(cur_obs_cartesian_position))


    # ✅ 관측 정보 저장
    state_total_np_dict['obs_cartesian_position'] = cur_obs_cartesian_position
    state_total_np_dict['obs_gripper_position'] = torch.from_numpy(np.array(cur_obs_gripper_pos))
    state_total_np_dict['obs_joint_positions'] = torch.from_numpy(np.array(cur_obs_joint_state))

    traj_len = min(len(cur_obs_image['1']), len(cur_obs_image['2']), len(cur_obs_image['wrist']))  # ✅ wrist 고려
    
    print('********** final process ***********')
    print(f'traj_len: {traj_len}')
    
    # ✅ 최종 행동 벡터 구성 (7차원 그대로)
    traj_actions = np.array(cur_actions)[:traj_len]
    print(f"traj_actions shape: {traj_actions.shape}")

    traj_qpos = np.concatenate((state_total_np_dict['obs_cartesian_position'], state_total_np_dict['obs_gripper_position']), axis=-1)[:traj_len]
    traj_qvel = np.zeros_like(traj_qpos)
    print(f"traj_qpos shape: {traj_qpos.shape}")
    print(f"traj_qvel shape: {traj_qvel.shape}")

    left_imgs = cur_obs_image['1'][:traj_len]
    right_imgs = cur_obs_image['2'][:traj_len]
    wrist_imgs = cur_obs_image['wrist'][:traj_len]  # ✅ wrist
    print(f"left_imgs shape: {left_imgs.shape}")
    print(f"right_imgs shape: {right_imgs.shape}")
    print(f"wrist_imgs shape: {wrist_imgs.shape}")

    # ✅ 관측 구성
    obs_replay = {
        'qpos': traj_qpos,
        'qvel': traj_qvel,
        'images': {
            'left': left_imgs,
            'right': right_imgs,
            'wrist': wrist_imgs  # ✅ wrist 포함
        }
    }
    cfg['lang_intrs'] = raw_lang
    generate_h5(obs_replay, traj_actions, cfg, total_traj_cnt, act_root_dir_path, edit_flag)
    total_traj_cnt += 1

