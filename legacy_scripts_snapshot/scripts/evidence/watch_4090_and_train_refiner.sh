#!/usr/bin/env bash
set -euo pipefail

# Monitor the physical RTX 4090 every 5 minutes. Once free VRAM is above the
# threshold, start prompt-refiner LoRA training in the background and exit.
#
# Important: this server's CUDA_VISIBLE_DEVICES mapping may differ from
# nvidia-smi physical GPU indices. By default we monitor physical GPU index 1
# (the RTX 4090 shown by nvidia-smi) but launch with CUDA_VISIBLE_DEVICES=0 as
# requested.

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/yyl/LLMMRA}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/yyl/miniconda3/envs/poi_data/bin/python}"

QUERY_GPU_INDEX="${QUERY_GPU_INDEX:-1}"
LAUNCH_CUDA_VISIBLE_DEVICES="${LAUNCH_CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-16384}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

RUN_NAME="${RUN_NAME:-prompt_refiner_lora_llama32_1b}"
OUTPUT_DIR="${OUTPUT_DIR:-models/prompt-refiner-lora-llama32-1b}"
LOG_DIR="${LOG_DIR:-runs/prompt_refiner_lora}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_NAME}.pid}"

BASE_MODEL="${BASE_MODEL:-/mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct}"
TRAIN_INPUTS="${TRAIN_INPUTS:-retrieval_assets/NewYork/distill/teacher_distill_inputs_train_3000.jsonl}"
TRAIN_OUTPUTS="${TRAIN_OUTPUTS:-retrieval_assets/NewYork/distill/teacher_prompt_outputs_train_3000.jsonl}"
VAL_INPUTS="${VAL_INPUTS:-retrieval_assets/NewYork/distill/teacher_distill_inputs_val_500.jsonl}"
VAL_OUTPUTS="${VAL_OUTPUTS:-retrieval_assets/NewYork/distill/teacher_prompt_outputs_val_500.jsonl}"

MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1536}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"

mkdir -p "${PROJECT_DIR}/${LOG_DIR}"
cd "${PROJECT_DIR}"

is_live_pid() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

find_existing_training_pid() {
  pgrep -u "$(id -u)" -f "scripts/evidence/train_prompt_refiner_lora.py" | head -n 1 || true
}

echo "[$(date '+%F %T')] monitor_start query_gpu_index=${QUERY_GPU_INDEX} launch_cuda_visible_devices=${LAUNCH_CUDA_VISIBLE_DEVICES} min_free_mb=${MIN_FREE_MB}" | tee -a "${LOG_FILE}"

while true; do
  gpu_rows="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>&1 || true)"
  if [[ -z "${gpu_rows}" ]]; then
    echo "[$(date '+%F %T')] ERROR: nvidia-smi returned no output" | tee -a "${LOG_FILE}"
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  gpu_row="$(printf '%s\n' "${gpu_rows}" | awk -F', ' -v idx="${QUERY_GPU_INDEX}" '$1 == idx {print $0}')"
  if [[ -z "${gpu_row}" ]]; then
    echo "[$(date '+%F %T')] ERROR: GPU index ${QUERY_GPU_INDEX} not found in nvidia-smi output: ${gpu_rows}" | tee -a "${LOG_FILE}"
    exit 1
  fi

  gpu_name="$(echo "${gpu_row}" | awk -F', ' '{print $2}')"
  mem_total="$(echo "${gpu_row}" | awk -F', ' '{print $3}')"
  mem_used="$(echo "${gpu_row}" | awk -F', ' '{print $4}')"
  mem_free="$(echo "${gpu_row}" | awk -F', ' '{print $5}')"
  echo "[$(date '+%F %T')] gpu=${QUERY_GPU_INDEX} name=${gpu_name} total_mb=${mem_total} used_mb=${mem_used} free_mb=${mem_free}" | tee -a "${LOG_FILE}"

  if [[ "${gpu_name}" != *"4090"* ]]; then
    echo "[$(date '+%F %T')] WARNING: monitored GPU name does not contain 4090: ${gpu_name}" | tee -a "${LOG_FILE}"
  fi

  if (( mem_free >= MIN_FREE_MB )); then
    existing_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if is_live_pid "${existing_pid}"; then
      echo "[$(date '+%F %T')] training_already_running pid=${existing_pid} source=pid_file" | tee -a "${LOG_FILE}"
      exit 0
    fi

    existing_pid="$(find_existing_training_pid)"
    if is_live_pid "${existing_pid}"; then
      echo "${existing_pid}" > "${PID_FILE}"
      echo "[$(date '+%F %T')] training_already_running pid=${existing_pid} source=process_scan pid_file=${PROJECT_DIR}/${PID_FILE}" | tee -a "${LOG_FILE}"
      exit 0
    fi

    echo "[$(date '+%F %T')] threshold_met launching_training" | tee -a "${LOG_FILE}"
    (
      export CUDA_VISIBLE_DEVICES="${LAUNCH_CUDA_VISIBLE_DEVICES}"
      export NCCL_P2P_DISABLE=1
      export NCCL_IB_DISABLE=1
      exec "${PYTHON_BIN}" scripts/evidence/train_prompt_refiner_lora.py \
        --train-inputs "${TRAIN_INPUTS}" \
        --train-outputs "${TRAIN_OUTPUTS}" \
        --val-inputs "${VAL_INPUTS}" \
        --val-outputs "${VAL_OUTPUTS}" \
        --base-model "${BASE_MODEL}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-source-length "${MAX_SOURCE_LENGTH}" \
        --max-target-length "${MAX_TARGET_LENGTH}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --bf16 \
        --gradient-checkpointing
    ) >> "${LOG_FILE}" 2>&1 &

    train_pid="$!"
    echo "${train_pid}" > "${PID_FILE}"
    echo "[$(date '+%F %T')] training_started pid=${train_pid} log=${PROJECT_DIR}/${LOG_FILE} pid_file=${PROJECT_DIR}/${PID_FILE}" | tee -a "${LOG_FILE}"
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
