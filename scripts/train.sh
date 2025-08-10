export CUDA_VISIBLE_DEVICES=0


#!/bin/bash

ACTION_HEAD=droid_diffusion # specify action policy head type
# define OUTPUT path

OUTPUT="/home/parkjeongsu/TinyVLA/OUTPUT_llava_pythia_3"

if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi
# backup the train scripts
cp ./scripts/train.sh $OUTPUT

# detailed usage of each parameter can be found in train_tinyvla.py
# 
#    --load_pretrain False \ ->  --load_pretrain True \ 
#  --model_pretrain /home/parkjeongsu/TinyVLA/Llava-Pythia-400M/model.safetensors \ 는 내가 추가한것

deepspeed --master_port 29600 --num_gpus=1 --num_nodes=1 ./train_tinyvla.py \
  --deepspeed /home/parkjeongsu/TinyVLA/llava-pythia/scripts/zero2.json \
  --lora_enable False \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
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
  --max_steps 3000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 50 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --action_head_type $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir /home/parkjeongsu/TinyVLA/Llava-Pythia-400M/log \
  --use_state False \
  #--window_size 6 \
 
 

for dir in "$OUTPUT"/*/ ; do
    # 'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp llava-pythia/preprocessor_config.json $dir
    fi
done

