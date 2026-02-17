#!/bin/bash

task_id=3
epoch=2
len=10

mkdir -p ./log

timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./log/output_mac_${task_id}_${timestamp}.log"

# ✅ ONLY A/B TESTING FLAGS FOR PERSONALIZATION SOURCES
python main_profile-slim-GNN.py \
    --model_path ../FlanT5-base/ \
    --emb_model_path ../bge-base-en-v1.5/ \
    --train_file ../LaMP_time_${task_id}_subset_id/train_aug_input.json \
    --dev_file ../LaMP_time_${task_id}_subset_id/dev_profile.json \
    --use_profile True \
    --use_session True \
    --use_graph True \
    --max_input_len 256 \
    --use_subset True \
    --max_his_len 512 \
    --max_session_len 7 \
    --max_new_len ${len} \
    --output_dir output_${task_id} \
    --optim adamw_torch \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_dir ./log/ \
    --report_to none \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --save_only_model True \
    --eval_steps 0.1 \
    --log_level warning \
    --save_total_limit 1 \
    --use_cpu True \
    2>&1 | tee "${LOG_FILE}"

echo "✅ Training log saved to ${LOG_FILE}"