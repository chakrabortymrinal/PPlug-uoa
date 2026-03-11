#!/bin/bash

set -u  # error on unset vars (but don't exit on command failure)

# --- Ensure conda env is active (works in non-interactive shells) ---
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pplug-env-m4}"

if ! command -v conda >/dev/null 2>&1; then
    echo "❌ conda not found on PATH. Start a conda shell first or set PYTHON_BIN."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV_NAME}"

# Use the env's python explicitly everywhere
PYTHON_BIN="$(which python)"
echo "✅ Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import sys; print('Python exe:', sys.executable); import transformers; print('transformers:', transformers.__version__)"
# --- End conda env activation ---

# ============================================================================
# ABLATION STUDY: Profile vs Session vs Graph Components
# ============================================================================

TASK_ID=3
EPOCHS=1
MAX_NEW_LEN=10
NUM_SEEDS=1
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"  # Set to 1 to keep all checkpoints, 0 to delete after metric extraction

# Output directories
BASE_OUTPUT="./ablation_results"
METRICS_DIR="${BASE_OUTPUT}/metrics"
CHECKPOINTS_DIR="${BASE_OUTPUT}/checkpoints"
LOGS_DIR="${BASE_OUTPUT}/logs"

mkdir -p "${METRICS_DIR}" "${CHECKPOINTS_DIR}" "${LOGS_DIR}"

# Timestamp for this ablation run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${BASE_OUTPUT}/ablation_summary_${TIMESTAMP}.json"
TABLE_FILE="${BASE_OUTPUT}/ablation_table_${TIMESTAMP}.md"

echo "🔬 Starting Ablation Study at ${TIMESTAMP}"
echo "📊 Results will be saved to: ${RESULTS_FILE}"
echo "📋 Table will be saved to: ${TABLE_FILE}"
echo ""

# ============================================================================
# EXPERIMENT CONFIGURATIONS (array-based, Bash 3+ compatible)
# ============================================================================

# VARIANT_NAMES=("baseline" "session" "graph" "full")
# VARIANT_PROFILES=("true" "true" "true" "true")
# VARIANT_SESSIONS=("false" "true" "false" "true")
# VARIANT_GRAPHS=("false" "false" "true" "true")
# # VARIANT_INST_TOKENS=("true" "true" "true" "true")

# ✅ Set to 1 to enable, 0 to disable
ENABLE_BASELINE=1
ENABLE_SESSION=0   # ✅ Disabled
ENABLE_GRAPH=1
ENABLE_FULL=1

# Build variant arrays dynamically
VARIANT_NAMES=()
VARIANT_PROFILES=()
VARIANT_SESSIONS=()
VARIANT_GRAPHS=()

if [ "${ENABLE_BASELINE}" -eq 1 ]; then
    VARIANT_NAMES+=("baseline")
    VARIANT_PROFILES+=("true")
    VARIANT_SESSIONS+=("false")
    VARIANT_GRAPHS+=("false")
fi

if [ "${ENABLE_SESSION}" -eq 1 ]; then
    VARIANT_NAMES+=("session")
    VARIANT_PROFILES+=("true")
    VARIANT_SESSIONS+=("true")
    VARIANT_GRAPHS+=("false")
fi

if [ "${ENABLE_GRAPH}" -eq 1 ]; then
    VARIANT_NAMES+=("graph")
    VARIANT_PROFILES+=("true")
    VARIANT_SESSIONS+=("false")
    VARIANT_GRAPHS+=("true")
fi

if [ "${ENABLE_FULL}" -eq 1 ]; then
    VARIANT_NAMES+=("full")
    VARIANT_PROFILES+=("true")
    VARIANT_SESSIONS+=("true")
    VARIANT_GRAPHS+=("true")
fi

# Validate at least one variant is enabled
if [ ${#VARIANT_NAMES[@]} -eq 0 ]; then
    echo "❌ Error: No variants enabled! Set at least one ENABLE_* flag to 1."
    exit 1
fi

echo "📋 Enabled variants: ${VARIANT_NAMES[*]}"
echo ""

# ============================================================================
# PROGRESS TRACKING (calculated after arrays are built)
# ============================================================================

TOTAL_RUNS=$(( ${#VARIANT_NAMES[@]} * NUM_SEEDS ))
RUNS_DONE=0
AB_START_TS=$(date +%s)

echo "🔧 Configuration:"
echo "   Task ID: ${TASK_ID}"
echo "   Epochs: ${EPOCHS}"
echo "   Seeds: ${NUM_SEEDS} (starting from 42)"
echo "   Variants: ${#VARIANT_NAMES[@]}"
echo ""

for i in "${!VARIANT_NAMES[@]}"; do
    echo "   [$i] ${VARIANT_NAMES[$i]}: profile=${VARIANT_PROFILES[$i]} session=${VARIANT_SESSIONS[$i]} graph=${VARIANT_GRAPHS[$i]}"
done

echo ""
echo "   Total experiments: ${TOTAL_RUNS}"
echo "   Checkpoints kept: $([ "${KEEP_CHECKPOINTS}" -eq 1 ] && echo "YES" || echo "NO")"
echo ""

if [ "${TOTAL_RUNS}" -gt 5 ]; then
    echo "⏱️  Estimated time: ~$((TOTAL_RUNS * 20)) minutes (assuming 20 min/run)"
    read -p "Press Enter to continue or Ctrl+C to abort..." 
    echo ""
fi

format_hms() {
    local total=$1
    local h=$((total / 3600))
    local m=$(((total % 3600) / 60))
    local s=$((total % 60))
    printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

print_progress() {
    local last_status="$1"
    local now elapsed avg eta pct
    now=$(date +%s)
    elapsed=$((now - AB_START_TS))

    if [ "${RUNS_DONE}" -gt 0 ]; then
        avg=$((elapsed / RUNS_DONE))
        eta=$((avg * (TOTAL_RUNS - RUNS_DONE)))
    else
        eta=0
    fi

    pct=$(( (RUNS_DONE * 100) / TOTAL_RUNS ))

    echo "📈 Progress: ${RUNS_DONE}/${TOTAL_RUNS} (${pct}%) | elapsed=$(format_hms "${elapsed}") | eta=$(format_hms "${eta}") | ${last_status}"
}

# ============================================================================
# FUNCTION: Run single training job
# ============================================================================

run_single_experiment() {
    local variant=$1
    local seed=$2
    local use_profile=$3
    local use_session=$4
    local use_graph=$5

    local exp_name="${variant}_seed${seed}"
    local output_dir="${CHECKPOINTS_DIR}/${exp_name}"
    local log_file="${LOGS_DIR}/${exp_name}.log"

    echo "▶️  Running: ${exp_name}"
    echo "   Profile=${use_profile} | Session=${use_session} | Graph=${use_graph}"

    start_time=$(date +%s)

    # Ensure relative paths inside Python resolve from extention/
    # and reduce thread/mem pressure + enable crash tracebacks.
    (
        cd "$(dirname "$0")" || { echo "❌ cd failed"; exit 2; }
        export PYTHONFAULTHANDLER=1
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export TOKENIZERS_PARALLELISM=false

        "${PYTHON_BIN}" main_profile-slim-GNN.py \
            --task_id ${TASK_ID} \
            --model_path ../FlanT5-large/ \
            --emb_model_path ../bge-base-en-v1.5/ \
            --train_file ../LaMP_time_${TASK_ID}_subset_id/train_aug_input.json \
            --dev_file ../LaMP_time_${TASK_ID}_subset_id/dev_profile.json \
            --use_profile ${use_profile} \
            --use_session ${use_session} \
            --use_graph ${use_graph} \
            --use_4bit False \
            --use_8bit False \
            --max_input_len 256 \
            --use_subset True \
            --max_his_len 512 \
            --max_session_len 8 \
            --max_new_len ${MAX_NEW_LEN} \
            --output_dir ${output_dir} \
            --optim adamw_torch \
            --learning_rate 1e-4 \
            --weight_decay 1e-4 \
            --warmup_ratio 0.05 \
            --num_train_epochs ${EPOCHS} \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --logging_dir ${LOGS_DIR}/${exp_name}/ \
            --logging_steps 10 \
            --evaluation_strategy steps \
            --save_strategy epoch \
            --save_only_model True \
            --eval_steps 0.1 \
            --log_level warning \
            --report_to none \
            --save_total_limit 1 \
            --seed ${seed} \
            --bf16 False \
            --use_cpu True \
            > "${log_file}" 2>&1
        exit $?
    ) || true

    exit_code=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ $exit_code -ne 0 ]; then
        echo "❌ FAILED: ${exp_name} (exit=${exit_code}) after $((duration / 60)) minutes"
        echo "   Log: ${log_file}"
        echo ""
        return $exit_code
    fi

    echo "✅ Completed in $((duration / 60)) minutes"
    echo ""

    extract_metrics "${output_dir}" "${exp_name}" "${duration}"

    if [ "${KEEP_CHECKPOINTS}" -ne 1 ]; then
        rm -rf "${output_dir}"
        echo "🧹 Deleted checkpoint dir: ${output_dir}"
        echo ""
    fi

    return 0
}

# ============================================================================
# FUNCTION: Extract metrics from checkpoint
# ============================================================================

extract_metrics() {
    local output_dir=$1
    local exp_name=$2
    local duration=$3

    local last_checkpoint
    last_checkpoint=$(ls -td ${output_dir}/checkpoint-* 2>/dev/null | head -1 || true)

    if [ -z "${last_checkpoint}" ]; then
        echo "⚠️  No checkpoint found for ${exp_name}"
        return 0
    fi

    if [ "${KEEP_CHECKPOINTS}" -eq 0 ]; then
        find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' ! -path "${last_checkpoint}" -exec rm -rf {} +
    fi

    local state_file="${last_checkpoint}/trainer_state.json"

    if [ ! -f "${state_file}" ]; then
        echo "⚠️  No trainer_state.json found in ${last_checkpoint}"
        return 0
    fi

    "${PYTHON_BIN}" "$(dirname "$0")/extract_metrics.py" \
      --state-file "${state_file}" \
      --exp-name "${exp_name}" \
      --duration-sec "${duration}" \
      --out-file "${METRICS_DIR}/${exp_name}.json"
}

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

echo "🧪 Running experiments..."
echo ""

failed=0

for i in "${!VARIANT_NAMES[@]}"; do
    variant="${VARIANT_NAMES[$i]}"
    use_profile="${VARIANT_PROFILES[$i]}"
    use_session="${VARIANT_SESSIONS[$i]}"
    use_graph="${VARIANT_GRAPHS[$i]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Variant: ${variant}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for seed in $(seq 42 $((42 + NUM_SEEDS - 1))); do
        if ! run_single_experiment "${variant}" "${seed}" "${use_profile}" "${use_session}" "${use_graph}"; then
            failed=$((failed + 1))
        fi
        RUNS_DONE=$((RUNS_DONE + 1))
        print_progress "${variant}_seed${seed}"
    done

    echo ""
done

# ============================================================================
# AGGREGATE RESULTS AND GENERATE TABLE
# ============================================================================

echo "📊 Aggregating results..."

"${PYTHON_BIN}" "$(dirname "$0")/aggregate_ablation_results.py" \
  --metrics-dir "${METRICS_DIR}" \
  --results-file "${RESULTS_FILE}" \
  --table-file "${TABLE_FILE}" \
  --timestamp "${TIMESTAMP}" \
  --num-seeds "${NUM_SEEDS}" \
  --task-id "${TASK_ID}"

# ============================================================================
# ✅ NEW: EVALUATE CHECKPOINTS FOR ACADEMIC PROOF
# ============================================================================

echo ""
echo "🔬 Running checkpoint evaluation for personalization effects..."
echo ""

EVAL_OUTPUT_FILE="${BASE_OUTPUT}/eval_results_${TIMESTAMP}.json"
EVAL_TABLE_FILE="${BASE_OUTPUT}/eval_comparison_${TIMESTAMP}.md"

"${PYTHON_BIN}" "$(dirname "$0")/evaluate_ablation_checkpoints.py" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --output-file "${EVAL_OUTPUT_FILE}" \
  --table-file "${EVAL_TABLE_FILE}" \
  --task-id "${TASK_ID}" \
  --max-samples 150

if [ -f "${EVAL_TABLE_FILE}" ]; then
    echo "✅ Checkpoint evaluation complete!"
    echo "📋 Comparison table:"
    cat "${EVAL_TABLE_FILE}"
else
    echo "⚠️  Evaluation table not generated"
fi

echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "✅ All experiments completed! (failed=${failed})"
echo "📊 View results:"
echo "   - Training metrics: ${RESULTS_FILE}"
echo "   - Training table: ${TABLE_FILE}"
echo "   - Checkpoint eval: ${EVAL_OUTPUT_FILE}"
echo "   - Comparison table: ${EVAL_TABLE_FILE}"
echo "   - Individual logs: ${LOGS_DIR}/"
echo ""