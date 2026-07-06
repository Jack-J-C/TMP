#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/users/yyl/TMP}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python}"

MONITOR_GPU_INDICES="${MONITOR_GPU_INDICES:-0,1}"
MIN_FREE_MB="${MIN_FREE_MB:-10240}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

RUN_NAME="${RUN_NAME:-poi_teamlora_reranker_anon3_compact_l1024_mainonly_step200_v1}"
OUTPUT_DIR_REL="${OUTPUT_DIR_REL:-models/poi-teamlora-reranker-anon3-compact-l1024-mainonly-step200-v1}"
TRAIN_LOG_REL="${TRAIN_LOG_REL:-logs/train_${RUN_NAME}.log}"
WATCH_LOG_REL="${WATCH_LOG_REL:-logs/watch_${RUN_NAME}.log}"
TRAIN_PID_FILE_REL="${TRAIN_PID_FILE_REL:-logs/train_${RUN_NAME}.pid}"
WATCH_PID_FILE_REL="${WATCH_PID_FILE_REL:-logs/watch_${RUN_NAME}.pid}"

TRAIN_SCRIPT_REL="${TRAIN_SCRIPT_REL:-src/poi_reranker/train_teamlora_reranker_raat.py}"
TRAIN_JOINED="${TRAIN_JOINED:-retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet}"
VAL_JOINED="${VAL_JOINED:-retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet}"
SEMANTIC_MAP="${SEMANTIC_MAP:-retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl}"
BASE_MODEL="${BASE_MODEL:-/mnt/data/users/yyl/TMP/models/Llama-3.2-1B-Instruct}"

mkdir -p "${PROJECT_DIR}/$(dirname "${TRAIN_LOG_REL}")"
mkdir -p "${PROJECT_DIR}/$(dirname "${WATCH_LOG_REL}")"
cd "${PROJECT_DIR}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

abs_path() {
  printf '%s/%s' "${PROJECT_DIR}" "${1}"
}

is_live_pid() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

find_existing_training_pid() {
  pgrep -a -u "$(id -u)" -f "${TRAIN_SCRIPT_REL}" \
    | awk -v marker="${OUTPUT_DIR_REL}" '$0 ~ marker {print $1; exit}'
}

launch_training() {
  local chosen_gpu="${1}"
  local chosen_name="${2}"
  local chosen_free_mb="${3}"
  local train_log
  local train_pid

  train_log="$(abs_path "${TRAIN_LOG_REL}")"
  log "threshold_met gpu=${chosen_gpu} name=${chosen_name} free_mb=${chosen_free_mb} launching_training"

  (
    export CUDA_VISIBLE_DEVICES="${chosen_gpu}"
    exec "${PYTHON_BIN}" "${TRAIN_SCRIPT_REL}" \
      --train-joined "${TRAIN_JOINED}" \
      --val-joined "${VAL_JOINED}" \
      --semantic-map "${SEMANTIC_MAP}" \
      --base-model "${BASE_MODEL}" \
      --output-dir "${OUTPUT_DIR_REL}" \
      --top-k 100 \
      --max-length 1024 \
      --expert-mode anonymous \
      --train-negatives 7 \
      --hard-negatives 6 \
      --batch-groups 1 \
      --grad-accum 8 \
      --max-steps 200 \
      --lr 2e-4 \
      --lora-r 8 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      --scorer-dropout 0.1 \
      --raat-mode none \
      --eval-steps 200 \
      --save-steps 200 \
      --bf16 \
      --gradient-checkpointing \
      --attn-implementation sdpa
  ) >> "${train_log}" 2>&1 &

  train_pid="$!"
  echo "${train_pid}" > "$(abs_path "${TRAIN_PID_FILE_REL}")"
  log "training_started pid=${train_pid} cuda_visible_devices=${chosen_gpu} train_log=${train_log} train_pid_file=$(abs_path "${TRAIN_PID_FILE_REL}")"
}

worker_loop() {
  local existing_pid
  local gpu_rows
  local raw_idx
  local idx
  local gpu_row
  local gpu_name
  local mem_total
  local mem_used
  local mem_free
  local chosen_idx=""
  local chosen_name=""
  local chosen_free=-1

  log "monitor_start monitor_gpu_indices=${MONITOR_GPU_INDICES} min_free_mb=${MIN_FREE_MB} check_interval_seconds=${CHECK_INTERVAL_SECONDS}"

  while true; do
    existing_pid="$(cat "$(abs_path "${TRAIN_PID_FILE_REL}")" 2>/dev/null || true)"
    if is_live_pid "${existing_pid}"; then
      log "training_already_running pid=${existing_pid} source=pid_file"
      exit 0
    fi

    existing_pid="$(find_existing_training_pid || true)"
    if is_live_pid "${existing_pid}"; then
      echo "${existing_pid}" > "$(abs_path "${TRAIN_PID_FILE_REL}")"
      log "training_already_running pid=${existing_pid} source=process_scan"
      exit 0
    fi

    gpu_rows="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>&1 || true)"
    if [[ -z "${gpu_rows}" ]]; then
      log "error nvidia-smi_returned_no_output"
      sleep "${CHECK_INTERVAL_SECONDS}"
      continue
    fi

    chosen_idx=""
    chosen_name=""
    chosen_free=-1

    IFS=',' read -r -a indices <<< "${MONITOR_GPU_INDICES}"
    for raw_idx in "${indices[@]}"; do
      idx="$(echo "${raw_idx}" | xargs)"
      gpu_row="$(printf '%s\n' "${gpu_rows}" | awk -F', ' -v target="${idx}" '$1 == target {print $0}')"

      if [[ -z "${gpu_row}" ]]; then
        log "warning gpu_index_not_found gpu=${idx}"
        continue
      fi

      gpu_name="$(echo "${gpu_row}" | awk -F', ' '{print $2}')"
      mem_total="$(echo "${gpu_row}" | awk -F', ' '{print $3}')"
      mem_used="$(echo "${gpu_row}" | awk -F', ' '{print $4}')"
      mem_free="$(echo "${gpu_row}" | awk -F', ' '{print $5}')"

      log "gpu=${idx} name=${gpu_name} total_mb=${mem_total} used_mb=${mem_used} free_mb=${mem_free}"

      if [[ "${gpu_name}" != *"5070 Ti"* ]]; then
        log "warning unexpected_gpu_name gpu=${idx} name=${gpu_name}"
      fi

      if (( mem_free >= MIN_FREE_MB && mem_free > chosen_free )); then
        chosen_idx="${idx}"
        chosen_name="${gpu_name}"
        chosen_free="${mem_free}"
      fi
    done

    if [[ -n "${chosen_idx}" ]]; then
      launch_training "${chosen_idx}" "${chosen_name}" "${chosen_free}"
      exit 0
    fi

    log "threshold_not_met sleeping_seconds=${CHECK_INTERVAL_SECONDS}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

start_daemon() {
  local watch_pid_file
  local watch_log
  local existing_pid
  local watch_pid

  watch_pid_file="$(abs_path "${WATCH_PID_FILE_REL}")"
  watch_log="$(abs_path "${WATCH_LOG_REL}")"
  existing_pid="$(cat "${watch_pid_file}" 2>/dev/null || true)"

  if is_live_pid "${existing_pid}"; then
    echo "watcher_already_running pid=${existing_pid} log=${watch_log}"
    exit 0
  fi

  setsid bash "$0" --worker >> "${watch_log}" 2>&1 < /dev/null &
  watch_pid="$!"
  echo "${watch_pid}" > "${watch_pid_file}"
  echo "watcher_started pid=${watch_pid} log=${watch_log} pid_file=${watch_pid_file}"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker_loop
else
  start_daemon
fi
