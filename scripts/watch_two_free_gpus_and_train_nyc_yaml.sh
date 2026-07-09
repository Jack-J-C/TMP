#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/users/yyl/TMP}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python}"

MONITOR_GPU_INDICES="${MONITOR_GPU_INDICES:-0,1}"
MIN_FREE_MB="${MIN_FREE_MB:-10240}"
REQUIRED_FREE_GPUS="${REQUIRED_FREE_GPUS:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

CONFIG_REL="${CONFIG_REL:-config/train_nyc_semprofile_simuser_v2.yaml}"
TRAIN_SCRIPT_REL="${TRAIN_SCRIPT_REL:-src/poi_reranker/train_teamlora_reranker_raat.py}"
RUN_NAME="${RUN_NAME:-train_nyc_semprofile_simuser_v2_yaml}"
TRAIN_LOG_REL="${TRAIN_LOG_REL:-logs/${RUN_NAME}.log}"
WATCH_LOG_REL="${WATCH_LOG_REL:-logs/watch_${RUN_NAME}.log}"
TRAIN_PID_FILE_REL="${TRAIN_PID_FILE_REL:-logs/${RUN_NAME}.pid}"
WATCH_PID_FILE_REL="${WATCH_PID_FILE_REL:-logs/watch_${RUN_NAME}.pid}"

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
    | awk -v cfg="${CONFIG_REL}" '$0 ~ cfg {print $1; exit}'
}

launch_training() {
  local chosen_gpu="${1}"
  local free_gpu_list="${2}"
  local train_log
  local train_pid

  train_log="$(abs_path "${TRAIN_LOG_REL}")"
  log "threshold_met free_gpus=${free_gpu_list} chosen_gpu=${chosen_gpu} min_free_mb=${MIN_FREE_MB} launching_training"

  (
    export CUDA_VISIBLE_DEVICES="${chosen_gpu}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    exec "${PYTHON_BIN}" "${TRAIN_SCRIPT_REL}" --config "${CONFIG_REL}"
  ) >> "${train_log}" 2>&1 &

  train_pid="$!"
  echo "${train_pid}" > "$(abs_path "${TRAIN_PID_FILE_REL}")"
  log "training_started pid=${train_pid} cuda_visible_devices=${chosen_gpu} config=${CONFIG_REL} train_log=${train_log}"
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
  local free_count
  local free_gpu_list
  local chosen_gpu
  local chosen_free

  log "monitor_start monitor_gpu_indices=${MONITOR_GPU_INDICES} required_free_gpus=${REQUIRED_FREE_GPUS} min_free_mb=${MIN_FREE_MB} check_interval_seconds=${CHECK_INTERVAL_SECONDS} config=${CONFIG_REL}"

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "error python_not_executable path=${PYTHON_BIN}"
    exit 1
  fi
  if [[ ! -f "${CONFIG_REL}" ]]; then
    log "error config_not_found path=${CONFIG_REL}"
    exit 1
  fi
  if [[ ! -f "${TRAIN_SCRIPT_REL}" ]]; then
    log "error train_script_not_found path=${TRAIN_SCRIPT_REL}"
    exit 1
  fi

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

    if ! gpu_rows="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>&1)"; then
      log "error nvidia-smi_failed output=${gpu_rows//$'\n'/; }"
      sleep "${CHECK_INTERVAL_SECONDS}"
      continue
    fi
    if [[ -z "${gpu_rows}" ]]; then
      log "error nvidia-smi_returned_no_output"
      sleep "${CHECK_INTERVAL_SECONDS}"
      continue
    fi

    free_count=0
    free_gpu_list=""
    chosen_gpu=""
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

      if [[ ! "${mem_free}" =~ ^[0-9]+$ ]]; then
        log "warning invalid_memory_free gpu=${idx} free_mb=${mem_free}"
        continue
      fi

      if (( mem_free >= MIN_FREE_MB )); then
        free_count=$((free_count + 1))
        free_gpu_list="${free_gpu_list}${free_gpu_list:+,}${idx}"
        if (( mem_free > chosen_free )); then
          chosen_gpu="${idx}"
          chosen_free="${mem_free}"
        fi
      fi
    done

    if (( free_count >= REQUIRED_FREE_GPUS )); then
      launch_training "${chosen_gpu}" "${free_gpu_list}"
      exit 0
    fi

    log "threshold_not_met free_count=${free_count}/${REQUIRED_FREE_GPUS} sleeping_seconds=${CHECK_INTERVAL_SECONDS}"
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
