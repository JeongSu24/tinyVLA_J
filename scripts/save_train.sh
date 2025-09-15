#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 메모리 단편화 완화 (권장)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

ACTION_HEAD=droid_diffusion

# OUTPUT 경로
OUTPUT="/home/parkjeongsu/TinyVLA/OUTPUT_llava_pythia_safe"

if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi

# 백업
cp ./scripts/train.sh $OUTPUT/train.sh.bak.$(date +%s)

# -------------------------
# 메모리-세이프 프리셋 요점
# - 이미지 224
# - 이미지 패치 토큰 비활성 (mm_use_im_patch_token=False)
# - batch_size 1 + GA 8  (유효 배치 8)
# - model_max_length 1024
# - dataloader workers 4
# - gradient checkpointing On
# -------------------------

deepspeed --master_port 29600 --num_gpus=1 --num_nodes=1 ./train_tinyvla.py \
  --deepspeed /home/parkjeongsu/TinyVLA/llava-pythia/scripts/zero2.json \
  --lora_enable False \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 224 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "droid_textEpisode" \
  --model_name_or_path /home/parkjeongsu/TinyVLA/Llava-Pythia-400M \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 30000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 30 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --action_head_type $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir /home/parkjeongsu/TinyVLA/Llava-Pythia-400M/log \
  --use_state True
