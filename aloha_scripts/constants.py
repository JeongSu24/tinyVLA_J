
##################### Setting of training data #####################################

# DATA_DIR = '/path/to/your/data_dir'

TASK_CONFIGS = {
    'droid_textEpisode': { # for local debug
        'dataset_dir': [
            "//home/parkjeongsu/TinyVLA/Droid/droid_with_lang/droid_1dot7t_lang_succ_t0001_s-0-0", # task 1
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'] # corresponding to image keys saved in h5py files
    },
}
####################################################################################

#!!!!!!!!!!!!!!!!!!!!!!Followings are copied from aloha which are not used!!!!!!!!!!!!!!!!!!!!!!
### ALOHA fixed constants
DT = 0.02

FPS = 50

PANDA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
]
START_ARM_POSE = [0, -0.785, 0, -2.356, 0, 1.57, 0.785]

# 그리퍼 위치(핑거 간 거리) 단위 : meter
GRIPPER_POSITION_OPEN = 0.04      # 완전히 펼친 상태(4cm)
GRIPPER_POSITION_CLOSE = 0.0      # 완전히 닫힌 상태(0cm)

# robosuite에서는 실제 action에 들어가는 값은 [left, right] 2개
GRIPPER_JOINT_OPEN = [0.02, 0.02]   # 양쪽 손가락 2cm 오픈
GRIPPER_JOINT_CLOSE = [0.0, 0.0]    # 완전히 닫힘

############################ Helper functions ############################


# 위치기반(폭) 정규화: 0(closed) ~ 1(open)
GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - GRIPPER_POSITION_CLOSE) / (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE)
GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE) + GRIPPER_POSITION_CLOSE

# Gripper joint 값 정규화 (좌/우 값은 항상 같으므로 하나만 사용가능)
GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - GRIPPER_JOINT_CLOSE[0]) / (GRIPPER_JOINT_OPEN[0] - GRIPPER_JOINT_CLOSE[0])
GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (GRIPPER_JOINT_OPEN[0] - GRIPPER_JOINT_CLOSE[0]) + GRIPPER_JOINT_CLOSE[0]
