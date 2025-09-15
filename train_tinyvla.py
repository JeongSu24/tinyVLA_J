import pickle
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"

from data_utils.datasets import load_data  # data functions
from data_utils.datasets import compute_dict_mean, set_seed  # helper functions

from aloha_scripts.constants_geminie import TASK_CONFIGS
from data_utils.datasets import LlavaPythiaProcess

import IPython
e = IPython.embed
import llava_pythia.llava_pythia_utils as LlavaUtils
from data_utils.processor import *
from llava_pythia.train.llava_pythia_trainer import LLaVAPythiaTrainer
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig

from dataclasses import dataclass, field, asdict

local_rank = None

#내가 추가한 Callback class

from transformers import TrainerCallback

class LossConsoleCallback(TrainerCallback):
    """
    Trainer가 주기적으로 내보내는 logs(dict)를 콘솔에 예쁘게 출력.
    - TrainerArguments.logging_steps 주기에 맞춰 on_log가 불려요.
    """
    def __init__(self, print_every: int = 1):
        self.print_every = print_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step) if state and state.global_step is not None else -1
        # logging_steps에 맞춰서만 출력 (안전장치)
        if step % max(1, self.print_every) != 0:
            return

        parts = [f"step={step}"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        # 선택: 평균 학습시간/속도 등도 logs에 있을 때 찍기
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.2f}")
        print(" | ".join(parts))


#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@dataclass
class ActionArguments:
    action_head_type: str = field(default="droid_diffusion") # action head type, 'act', 'droid_diffusion'
    action_dim: int = field(default=10)
    state_dim: int = field(default=7)
    chunk_size: int = field(default=16) # size of action chunk, same as mobile aloha
     # 추가
    use_state: bool = field(default=False, metadata={"help": "Whether to use robot state as input to the policy"})
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # equals to base model path when set load_pretrain=True
    version: Optional[str] = field(default="v0")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)

    concat: str = field(default="None")

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    task_name: str = field(default="example_task_config")
    skip_mirrored_data: bool = field(default=True)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)
    remove_unused_columns: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    logging_dir: str = field(default='./logs')
    logging_strategy: str = field(default='steps')
    logging_steps: int = field(default=10)

    save_steps: int = field(default=10)  # interval for saving checkpoint
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=5000) # maxmium training steps
    seed: int = field(default=0)

    # validate
    do_eval: bool = field(default=False) # unused
    evaluation_strategy: str = field(default="steps") # unused
    eval_steps: int = field(default=200) # unused
    per_device_eval_batch_size: int = field(default=32) # batch size per device

    # pretrain
    load_pretrain: bool = False # unused
    pretrain_image_size : int = 320 # default 320 x 180

    dataloader_pin_memory: bool = False
    # lora
    lora_enable: bool = False # specify using lora or not
    lora_module: str = "vit llm" # specify which part to use lora, separated by spaces
    lora_r: int = 64
    lora_task_type: str = 'CAUSAL_LM'#'FEATURE_EXTRACTION'
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    non_lora_lr: Optional[float] = 3e-5 # learning rate for non lora part: Diffusion Policy head;
    # learning_rate is inherited from parent class, used for lora part
    group_by_modality_length: bool = field(default=False)

    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def parse_pythia():
    """
    Parses command-line arguments into dataclasses for model, data, training, and action configurations.

    This function uses the HfArgumentParser to parse command-line arguments into structured dataclasses.
    It sets the global `local_rank` variable based on the training arguments and configures quantization
    settings if specified in the training arguments.

    Returns:
        tuple: A tuple containing the parsed model, data, training, and action arguments, 
               a configuration object for the LlavaPythia model, and a dictionary for 
               model loading arguments with quantization settings.
               ✅ 마지막에 use_state도 함께 반환함 (config 등에서 직접 접근 가능)
    """

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionArguments))
    model_args, data_args, training_args, action_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    config = LlavaPythiaConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # add parameters about action head
    for k in asdict(action_args).keys():
        setattr(config, k, getattr(action_args, k))
    config.concat = model_args.concat

    # ✅ use_state도 반환에 포함되도록 추가 (config 등에서 활용 가능)
    return model_args, data_args, training_args, action_args, config, bnb_model_from_pretrained_args, action_args.use_state


def train_bc(train_dataset=None, val_dataset=None, model=None, config=None, sampler_params=None, tokenizer=None):
    """
    Args:
        train_dataset: The dataset used for training.
        val_dataset: The dataset used for validation.
        model: The model to be trained.
        config: Configuration dictionary containing training arguments.
        sampler_params: Parameters for the data sampler.
        tokenizer: Tokenizer used for processing the input data.
    """
    set_seed(config['training_args'].seed)

    
    #data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer) 아래 코드로 수정: state_use가 True, False 인걸 받으려고
    data_collator = DataCollatorForSupervisedDataset(
    tokenizer=tokenizer,
    use_state=config['data_args'].use_state,          # ← 중요
    state_dim=config['action_args'].state_dim         # 보통 7
)
    data_module = dict(train_dataset=train_dataset,
                       data_collator=data_collator,
                       eval_dataset=val_dataset
                       )
    trainer = LLaVAPythiaTrainer(model=model,
                                 tokenizer=tokenizer,
                                 args=config['training_args'],
                                 sampler_params=sampler_params,
                                 callbacks=[LossConsoleCallback(print_every=config['training_args'].logging_steps)],
                                 **data_module)
    
    #--mm_use_im_patch_token True가 잘 작동되는지
    
    trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if config['training_args'].lora_enable:
        state_dict = LlavaUtils.get_peft_state_maybe_zero_3(
            model.named_parameters(), config['training_args'].lora_bias
        )
        non_lora_state_dict = LlavaUtils.get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if config['training_args'].local_rank == 0 or config['training_args'].local_rank == -1:
            model.config.save_pretrained(config['training_args'].output_dir)
            model.save_pretrained(config['training_args'].output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict,
                       os.path.join(config['training_args'].output_dir, 'non_lora_trainables.bin'))
    else:
        LlavaUtils.safe_save_model_for_hf_trainer(trainer=trainer,
                                                  output_dir=config['training_args'].output_dir)


# === 디버깅용: forward 감싸서 states 확인 ===
def wrap_forward_with_inspect(model):
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        print(f"\n✅ Wrapped forward called")
        print(f"🔸 kwargs keys: {list(kwargs.keys())}")

        if 'states' in kwargs:
            if kwargs['states'] is None:
                print("   ⚠ 'states' is None")
            elif isinstance(kwargs['states'], torch.Tensor):
                print(f"   🔹 states shape: {kwargs['states'].shape}")
            else:
                print(f"   ⚠ 'states' exists but is not tensor: {type(kwargs['states'])}")
        else:
            print("   ❌ 'states' not found in kwargs")

        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward


def main(config=None, llava_pythia_config=None):
    """
    Main function to set up and execute the training process.

    This function initializes the tokenizer and model based on the provided configuration.
    It loads the training and validation datasets and calls the `train_bc` function to perform
    the training. After training, it saves dataset statistics.

    Args:
        config: Configuration dictionary containing model, data, training, and action arguments.
        llava_pythia_config: Configuration object for the LlavaPythia model.
    """
    set_seed(1)
    # command line parameters
    training_args = config['training_args'].__dict__
    # get task parameters
    task_config = TASK_CONFIGS[config['data_args'].task_name]
    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.95)
    name_filter = task_config.get('name_filter', lambda n: True)

    config['camera_names'] = camera_names
    config['episode_len'] = episode_len

    if 'pythia' in config['model_args'].model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config['model_args'].model_name_or_path,
            cache_dir=config['training_args'].cache_dir,
            model_max_length=config['training_args'].model_max_length,
            padding_side="right"
        )
        tokenizer.pad_token_id = 1

    # ✅ action_args의 use_state 값을 data_args로 복사
    config['data_args'].use_state = config['action_args'].use_state
    
    model, data_args = LlavaUtils.load_llava_pythia(config=config, llava_pythia_config=llava_pythia_config, rank0_print=rank0_print, tokenizer=tokenizer)
    
    #use_state 디버깅용 함수
    wrap_forward_with_inspect(model)


    # prepare process class
    llava_pythia_process = LlavaPythiaProcess(data_args, tokenizer=tokenizer)

    # load data
    train_dataset, val_dataset, stats, sampler_params = load_data(dataset_dir, name_filter, camera_names, config['training_args'].per_device_train_batch_size,
                                                           config['training_args'].per_device_eval_batch_size, config['action_args'].chunk_size,
                                                           skip_mirrored_data=config['data_args'].skip_mirrored_data,
                                                           config=config,
                                                           policy_class=config['action_args'].action_head_type, stats_dir_l=stats_dir,
                                                           sample_weights=sample_weights, train_ratio=train_ratio, return_dataset=True, llava_pythia_process=llava_pythia_process,
                                                            use_state=config['data_args'].use_state)   # ✅ 추가 : use_state=config['data_args'].use_state

    best_ckpt_info = train_bc(train_dataset=train_dataset, model=model, val_dataset=val_dataset, config=config, sampler_params=sampler_params, tokenizer=tokenizer)
    # save dataset stats
    stats_path = os.path.join(config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)


if __name__ == '__main__':
    # parse args
    model_args, data_args, training_args, action_args, llava_pythia_config, bnb_model_from_pretrained_args, use_state = parse_pythia() #use_state추가
    config = {
        'model_args':model_args,
        'data_args':data_args,
        'training_args':training_args,
        'action_args': action_args,
        'bnb_model_from_pretrained_args':bnb_model_from_pretrained_args
    }
    import json
    config_dict = {k:asdict(v) if not isinstance(v, dict) else v for k,v in config.items()}

    main(config=config, llava_pythia_config=llava_pythia_config)
    pass


