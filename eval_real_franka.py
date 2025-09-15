# =============================================================
# Stage 1) Imports & Utility Functions (import 및 유틸 함수 주석)
# =============================================================

import os
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch
from torchvision import transforms
import cv2
from copy import deepcopy
from itertools import repeat
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import numpy as np
import time
from aloha_scripts.constants_geminie import FPS

from data_utils.datasets import set_seed
from llava_pythia.model import *
from einops import rearrange
import torch_utils as TorchUtils
import matplotlib.pyplot as plt
import sys


# -------------------------------------------------------------
# 함수: get_image
# 목적: 환경 관측 ts(transition)에서 지정된 카메라들의 이미지를 가져와 텐서로 변환
# 입력:
#   - ts: 환경에서 반환한 transition/state 객체(여기서는 ts.observation['images'][cam_name] 접근)
#   - camera_names: 사용할 카메라 이름 리스트 (예: ["left", "right"])
#   - rand_crop_resize: True면 랜덤 크롭 후 원래 크기로 리사이즈 (데이터 증강 목적)
# 출력:
#   - shape: (1, C_view, C, H, W) 형태의 CUDA float 텐서 (0~1 스케일)
# 비고: LLaVA-Pythia의 이미지 프로세서 입력을 고려해 배치 차원을 앞에 둠
# -------------------------------------------------------------
def get_image(ts, camera_names, rand_crop_resize=False):
    """
    Retrieves and processes images from the specified cameras.

    Args:
        ts: The timestamp or data structure containing observations.
        camera_names: List of camera names to retrieve images from.
        rand_crop_resize: Boolean indicating whether to apply random crop and resize.

    Returns:
        A tensor containing the processed images.
    """
    curr_images = []
    for cam_name in camera_names:
        # ts.observation['images'][cam_name]: HWC → CHW 로 재배열
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)

    # 뷰(카메라) 차원을 쌓음: (N_view, C, H, W)
    curr_image = np.stack(curr_images, axis=0)
    # [0,255] → [0,1] 정규화 + torch 텐서 변환 + 배치 차원 추가: (1, N_view, C, H, W)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        # 간단한 중앙 크롭(비율 사용) 후, 원 크기로 리사이즈
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        # (1, N_view, C, H, W) → (N_view, C, H, W)로 임시 축소 이후 Resize
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        # 다시 배치 차원 추가
        curr_image = curr_image.unsqueeze(0)
    return curr_image


# -------------------------------------------------------------
# 함수: pre_process
# 목적: 로봇 상태 스칼라/벡터를 통계량(mean/std)로 정규화
# 입력:
#   - robot_state_value: 정규화할 값 (numpy 혹은 torch 텐서 가정)
#   - key: 통계 사전에서 참조할 이름 접두어
#   - stats: {key+'_mean', key+'_std'}를 포함하는 dict (pickle로 로드된 dataset_stats)
# 출력:
#   - 정규화된 값
# -------------------------------------------------------------
def pre_process(robot_state_value, key, stats):
    """
    Pre-processes the robot state value using provided statistics.

    Args:
        robot_state_value: The raw robot state value.
        key: The key to access the corresponding statistics.
        stats: Dictionary containing mean and standard deviation for normalization.

    Returns:
        The normalized robot state value.
    """
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


# -------------------------------------------------------------
# 함수: get_obs
# 목적: 실제 배포 환경에서 관측(이미지, 상태)을 수집하여 모델 입력 형태로 반환
# 현재는 예시용 스텁 함수. 사용자 환경에 맞게 구현 필요
# 출력:
#   - images, states (둘 다 모델 입력 규격)
# -------------------------------------------------------------
def get_obs():
    """
    Retrieves observations (images and robot states) from the robot environment.

    Returns:
        A tuple containing images and states.
    """
    return None, None # images, states


# -------------------------------------------------------------
# 함수: time_ms
# 목적: 현재 시간을 millisecond 단위로 반환 (로깅/지연 제어 등에 사용)
# -------------------------------------------------------------
def time_ms():
    return time.time_ns() // 1_000_000


# -------------------------------------------------------------
# 함수: convert_actions
# 목적: 모델이 출력한 행동 벡터(10D: xyz(3)+rot6d(6)+grip(1))를
#       환경에 필요한 (xyz + Euler(3) + gripper) 형태로 변환
# 입력:
#   - pred_action: numpy 1D, 길이 10 (모델 출력 after post_process)
# 출력:
#   - 변환된 행동: (xyz 3 + euler 3 + gripper 1) → 총 7D
# 주의:
#   - rot_6d_to_euler_angles는 TorchUtils에 의존
#   - 환경에 따라 Euler convention("XYZ")를 일치시켜야 함
# -------------------------------------------------------------

def convert_actions(pred_action):
    # pred_action: 10D [xyz(3), rot6d(6), grip(1)]
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    # 6D 회전을 Euler 각으로 복원
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()

    # (xyz, euler, gripper)로 재구성
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    print(f'4. after convert pred_action: {pred_action}')

    return pred_action


# =============================================================
# Stage 2) Policy Class (policy 클래스 주석)
# =============================================================

class llava_pythia_act_policy:
    """
    LLaVA-Pythia 기반 액션 폴리시 클래스.

    역할:
      - 사전학습 VLM(vision-language model) + 학습된 어댑터/LoRA를 로드
      - 입력(이미지, 상태, 텍스트 프롬프트)을 모델이 요구하는 포맷으로 가공
      - forward 호출 시 적절한 키워드 인자를 구성하여 policy(policy.policy)를 실행

    주요 속성:
      - policy_config: 모델 경로, 대화 템플릿, LoRA 사용 여부, action head 종류 등 설정 dict
      - tokenizer, policy, image_processor, context_len: load_pretrained_model에서 로드된 객체들
      - config: 모델 구성(예: action_dim, chunk_size 등)을 담은 구성 객체
      - conv: 대화 템플릿 (이미지 토큰 + 사용자 질의/지시문을 포함)
    """

    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    # ---------------------------------------------------------
    # 메서드: load_policy
    # 목적: 주어진 설정에 따라 사전학습/LoRA 병합 모델을 메모리에 로드
    # 동작:
    #   - 모델 이름 추출(get_model_name_from_path)
    #   - load_pretrained_model 호출로 tokenizer, model(policy), image_processor 등 로드
    #   - 구성(config)을 파일 경로에서 읽어서 (예: action_dim, chunk_size) 사용
    # ---------------------------------------------------------
    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        # tokenizer, policy(model), image_processor, context_len 로드
        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, False, False
        )

        # 구성 로드 (trust_remote_code=True: 커스텀 구성이 허용됨)
        self.config = LlavaPythiaConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    # ---------------------------------------------------------
    # 메서드: process_batch_to_llava
    # 목적: (이미지, 로봇상태, 텍스트)을 LLaVA-Pythia forward에 바로 넣을 수 있는 dict로 변환
    # 입력:
    #   - curr_image: (N_view, C, H, W) 혹은 (1, N_view, C, H, W) 텐서
    #   - robo_state: (B, state_dim) 텐서 (정규화된 상태가 권장)
    #   - raw_lang: 자연어 지시문 문자열
    # 출력:
    #   - dict(input_ids, attention_mask, images, images_r, states)
    # 절차:
    #   1) conv 템플릿 준비, 이미지 토큰/문장 토큰 결합 프롬프트 생성
    #   2) 멀티뷰 이미지 분할(image, image_r) 후 image_processor로 전처리
    #   3) tokenizer로 프롬프트를 토크나이즈, 마스크 생성
    #   4) states(로봇 상태)를 올바른 dtype/device로 올림
    # ---------------------------------------------------------
    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Processes a batch of data for Llava-Pythia model input.

        Args:
            curr_image: Current image tensor.
            robo_state: Current robot state tensor.
            raw_lang: Raw language input.

        Returns:
            A dictionary containing processed data for the model.
        """
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        # 입력 이미지 텐서 크기 정리 (모델은 (N_view, C, H, W) 기대)
        if len(curr_image.shape) == 5: # (1, N_view, C, H, W)
            curr_image = curr_image.squeeze(0)

        # 두 개의 뷰로 분리: (N_view=2 가정) image, image_r
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        # 정사각형 패딩(검은 바탕) → image_processor 전처리
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values']
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # 텍스트 프롬프트 생성: 이미지 토큰 + 사용자 지시문
        inp = raw_lang
        assert image is not None, 'image must be provided.'
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        # 토크나이징 및 마스크 생성
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 로봇 상태를 모델 dtype/device로 이동
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        # 모델 호출에 필요한 키 구성
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states
        )
        return data_dict

    # ---------------------------------------------------------
    # 메서드: expand2square
    # 목적: (B, C, H, W) 이미지를 정사각형 캔버스(B, H_max, W_max, C)에 중앙 정렬 패딩
    #       이후 torch 텐서로 되돌림(B, H_max, W_max, C) — image_processor가 원하는 포맷 맞춤
    # 주의: dtype/device를 원본과 동일하게 유지
    # ---------------------------------------------------------
    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        # numpy 캔버스 초기화: (B, Hmax, Wmax, C)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

        if height == width:
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device) # (B, H, W, C)
        return expanded_imgs


# =============================================================
# Stage 3) eval loop (eval_bc 실행 주석)
# =============================================================

# -------------------------------------------------------------
# 함수: eval_bc
# 목적: 배치된 환경(deploy_env)에서 정책(policy)을 평가(rollout)하며 액션을 질의/집계/후처리
# 입력:
#   - policy: llava_pythia_act_policy 인스턴스(내부에 tokenizer/policy/image_processor 포함)
#   - deploy_env: 사용자가 구현한 환경 객체 (reset, get_observation, step 메서드 필요)
#   - policy_config: 설정 dict (action_head, action_head_type 등)
#   - save_episode: 에피소드 저장 여부(현재 코드에서는 시각화/저장 부분은 생략 상태)
#   - num_rollouts: 롤아웃 횟수
#   - raw_lang: 텍스트 지시문 (필수)
# 핵심 로직:
#   1) 통계(stats) 로드 후 post_process 정의 (학습 시 정규화의 역변환)
#   2) temporal aggregation: chunk_size 길이만큼 예측을 모아 지수 가중으로 집계
#   3) all_time_actions: (T_max, T_max+num_queries, action_dim) 캐시로 sliding aggregation 수행
#   4) 정책 추론 결과 raw_action → post_process → convert_actions → env.step
# -------------------------------------------------------------

def eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=None):
    """
    Evaluates the behavior cloning policy in the deployment environment.

    Args:
        policy: The policy to evaluate.
        deploy_env: The deployment environment.
        policy_config: Configuration dictionary for the policy.
        save_episode: Whether to save the episode data.
        num_rollouts: Number of rollouts to perform.
        raw_lang: Raw language input for the policy.

    Returns:
        None
    """
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)  # 재현성 고정

    # action_head에 따라 전처리/집계 방식 토글
    if policy_config["action_head"] == 'act':
        rand_crop_resize = False
        temporal_agg = True
    else:
        rand_crop_resize = True
        temporal_agg = True

    action_dim = policy.config['action_dim']

    # 모델을 eval 모드로 전환
    policy.policy.eval()

    # 데이터셋 통계 로드(정규화 역변환용)
    import pickle
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 예측 후처리 함수 선택 (학습 시 스케일링 복원)
    if policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    env = deploy_env

    # 질의 주기 및 집계 길이 설정
    query_frequency = policy.config['chunk_size'] / 2  # 실행 간격 (chunk_size보다 작아야 함)
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config['chunk_size']

    max_timesteps = int(10000)  # 실제 로봇이면 더 크게 필요할 수도 있음

    for rollout_id in range(num_rollouts):
        rollout_id += 0
        env.reset(randomize=False)
        print(f"env has reset!")

        # temporal aggregation 캐시: 시간 t에 대해 t..t+num_queries 범위의 예측을 저장
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim], dtype=torch.float16).cuda()

        image_list = []   # 시각화 용 (현재는 누적만; 별도 저장 로직은 사용자 환경 의존)
        robot_state_list = []
        target_action_list = []

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS  # 주기 제어(루프 타임)
            culmulated_delay = 0
            for t in range(max_timesteps):

                # 사용자 환경에서 관측 수집
                obs = deploy_env.get_observation()

                # 사용자 정의 get_obs: (이미지 numpy, 상태 numpy)로 변환한다고 가정
                traj_rgb_np, robot_state = get_obs(obs, stats)
                image_list.append(traj_rgb_np)

                # 상태 텐서화
                robot_state = torch.from_numpy(robot_state).float().cuda()

                # query_frequency에 맞춰 이미지 텐서 준비(증강 포함)
                if t % query_frequency == 0:
                    curr_image = torch.from_numpy(traj_rgb_np / 255.0).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                # 초기 워밍업(텐서 RT 속도 안정화)
                if t == 0:
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        policy.policy(**batch, eval=True)
                    print('network warm up done')
                    time1 = time.time()

                # === 정책 질의 ===
                if policy_config['action_head_type'] == "act":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        # 지수 가중 평균(최근 예측에 더 높은 가중치)
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif policy_config['action_head_type'] == "droid_diffusion":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                print(f"raw action size: {raw_action.size()}")

                # === 후처리(post_process) 및 회전형식 변환 ===
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")

                # rot6d → euler 로 변환하여 환경 제어 명령으로 사용
                action = convert_actions(action)

                # === 환경 스텝 ===
                print(f'step {t}, pred action: {action}')
                action_info = deploy_env.step(action)

                # === 로깅/타이밍 ===
                robot_state_list.append(robot_state)
                target_action_list.append(action)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(
                        f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()

    return


# -------------------------------------------------------------
# 실행 엔트리: __main__
# 목적: 정책 생성 및 평가 루프 호출. 실제 사용 시 model_path/model_base, deploy_env를 사용자 환경에 맞게 설정 필요.
# -------------------------------------------------------------
if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 하이퍼파라미터 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'droid_diffusion'  # specify the action head type
    policy_config = {
        "model_path": f"/path/to/trained/VLA",   # 학습된(혹은 LoRA 적용된) 모델 가중치 경로
        "model_base": f"/path/to/pretrained/VLM", # LoRA merge 시 사용하는 base VLM 경로
        "enable_lora": True,                      # LoRA 사용 여부
        "conv_mode": "pythia",                   # 대화 템플릿 모드
        "action_head": action_head,               # action head 유형: 'act' | 'transformer_diffusion' | 'droid_diffusion'
    }
    global im_size
    im_size = 320  # 입력 이미지 기본 크기(모델 전처리기에 맞춰 조정 가능)
    raw_lang = 'put the tennis ball on the right side into the tennis bucket'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # 정책 객체 생성 (모델/토크나이저/프로세서 로드)
    policy = llava_pythia_act_policy(policy_config)

    ############################################################################################################
    # 사용자의 실제 로봇/시뮬레이터 환경 객체를 준비해야 합니다.
    # 반드시 다음 메서드가 구현되어야 합니다:
    #   - reset(randomize: bool) -> None
    #   - get_observation() -> 사용자 정의 관측 구조체
    #   - step(action: np.ndarray) -> Any (환경 피드백)
    deploy_env = None
    ############################################################################################################

    # 평가 루프 호출 (raw_lang은 필수)
    eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang)
