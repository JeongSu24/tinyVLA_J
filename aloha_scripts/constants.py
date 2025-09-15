
##################### Setting of training data #####################################

# DATA_DIR = '/path/to/your/data_dir'

TASK_CONFIGS = {
    'droid_textEpisode': { # for local debug
        'dataset_dir': [
            "/home/parkjeongsu/TinyVLA/Droid/trans_data/droid_1dot7t_lang_succ_t0001_s-0-0", # task 1
        ],
        # 한 에피소드 길이(스텝 수). DROID는 150~300 내외가 흔하므로 200으로 시작하고,
        # 로더 통계 출력(stats)로 확인 후 조정하세요.
        'episode_len': 200,  # 1000,
        'camera_names': ['left', 'right', 'wrist'] # corresponding to image keys saved in h5py files
    },
}
####################################################################################

#!!!!!!!!!!!!!!!!!!!!!!Followings are copied from aloha which are not used!!!!!!!!!!!!!!!!!!!!!!