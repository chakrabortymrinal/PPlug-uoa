#!/bin/bash

task_id=3
epoch=8
len=10

mkdir -p ./log

timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./log/output_mac_${task_id}_${timestamp}.log"
TENSORBOARD_DIR="./log/runs_${task_id}_${timestamp}"

# 🔥 Kill any existing TensorBoard processes on port 6006
lsof -ti:6006 | xargs kill -9 2>/dev/null || true

# 🔥 Start TensorBoard in background
echo "📊 Starting TensorBoard..."
tensorboard --logdir="${TENSORBOARD_DIR}" --port=6006 --host=127.0.0.1 > /dev/null 2>&1 &
TENSORBOARD_PID=$!

sleep 2

echo "============================================================"
echo "📊 TensorBoard started at http://127.0.0.1:6006/"
echo "   Process ID: ${TENSORBOARD_PID}"
echo "   Log directory: ${TENSORBOARD_DIR}"
echo "============================================================"
echo ""

# ✅ FIXED: Smoother evaluation frequency
python main_profile-slim-GNN.py \
    --model_path ../FlanT5-large/ \
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
    --logging_dir "${TENSORBOARD_DIR}" \
    --report_to tensorboard \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_only_model False \
    --log_level warning \
    --save_total_limit 2 \
    --use_cpu True \
    2>&1 | tee "${LOG_FILE}"

# 🛑 Stop TensorBoard after training
echo ""
echo "============================================================"
echo "✅ Training complete!"
echo "✅ Training log saved to ${LOG_FILE}"
echo "📊 TensorBoard logs saved to ${TENSORBOARD_DIR}"
echo ""
echo "🛑 Stopping TensorBoard (PID: ${TENSORBOARD_PID})..."
kill ${TENSORBOARD_PID} 2>/dev/null && echo "   TensorBoard stopped." || echo "   TensorBoard already stopped."
echo ""
echo "💡 To view logs later, run:"
echo "   tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo "============================================================"