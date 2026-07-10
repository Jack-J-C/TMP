#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/users/yyl/TMP}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python}"
CLSPREC_REPO="${CLSPREC_REPO:-/mnt/data/users/yyl/CLSPRec}"
CLSPREC_DATA_DIR="${CLSPREC_DATA_DIR:-/mnt/data/users/yyl/CLSPRec/processed_data/tmp_nyc}"

GPU_INDEX="${GPU_INDEX:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-10240}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-TMP_NYC_CLSPRec}"

LOG_DIR="${PROJECT_DIR}/logs"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_gpu0_clsprec_nyc.log}"
WATCH_PID_FILE="${WATCH_PID_FILE:-${LOG_DIR}/watch_gpu0_clsprec_nyc.pid}"
RUN_INFO_FILE="${RUN_INFO_FILE:-${LOG_DIR}/watch_gpu0_clsprec_nyc.last_run}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

is_live_pid() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

cleanup_pid() {
  local current
  current="$(cat "${WATCH_PID_FILE}" 2>/dev/null || true)"
  if [[ "${current}" == "$$" ]]; then
    rm -f "${WATCH_PID_FILE}"
  fi
}

validate_inputs() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "error python_not_executable path=${PYTHON_BIN}"
    exit 1
  fi
  if [[ ! -f "${PROJECT_DIR}/scripts/run_clsprec_tmp_nyc.py" ]]; then
    log "error runner_not_found path=${PROJECT_DIR}/scripts/run_clsprec_tmp_nyc.py"
    exit 1
  fi
  if [[ ! -d "${CLSPREC_REPO}" ]]; then
    log "error clsprec_repo_not_found path=${CLSPREC_REPO}"
    exit 1
  fi
  if [[ ! -d "${CLSPREC_DATA_DIR}" ]]; then
    log "error clsprec_data_dir_not_found path=${CLSPREC_DATA_DIR}"
    exit 1
  fi
}

free_memory_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null \
    | awk 'NR==1 {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}'
}

launch_clsprec() {
  local run_name
  local train_log

  run_name="${RUN_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S)"
  train_log="${LOG_DIR}/train_eval_clsprec_nyc_${run_name}.log"

  {
    echo "run_name=${run_name}"
    echo "gpu_index=${GPU_INDEX}"
    echo "train_log=${train_log}"
    echo "metrics_json=${CLSPREC_REPO}/results/${run_name}_test_metrics.json"
    echo "started_at=$(timestamp)"
  } > "${RUN_INFO_FILE}"

  log "launch_clsprec gpu=${GPU_INDEX} run_name=${run_name} log=${train_log}"
  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" scripts/run_clsprec_tmp_nyc.py \
    --repo "${CLSPREC_REPO}" \
    --data-dir "${CLSPREC_DATA_DIR}" \
    --city NYC \
    --gpu cuda:0 \
    --epoch 25 \
    --run-name "${run_name}" \
    > "${train_log}" 2>&1

  log "clsprec_finished run_name=${run_name} metrics=${CLSPREC_REPO}/results/${run_name}_test_metrics.json"
}

main() {
  local existing_pid
  local free_mb

  existing_pid="$(cat "${WATCH_PID_FILE}" 2>/dev/null || true)"
  if is_live_pid "${existing_pid}"; then
    echo "watcher_already_running pid=${existing_pid} log=${WATCH_LOG}"
    exit 0
  fi
  echo "$$" > "${WATCH_PID_FILE}"
  trap cleanup_pid EXIT

  validate_inputs
  log "monitor_start task=CLSPRec_NYC gpu=${GPU_INDEX} min_free_mb=${MIN_FREE_MB} check_interval_seconds=${CHECK_INTERVAL_SECONDS}"

  while true; do
    free_mb="$(free_memory_mb || true)"
    if [[ ! "${free_mb}" =~ ^[0-9]+$ ]]; then
      log "warning cannot_read_gpu_free_memory gpu=${GPU_INDEX} value=${free_mb:-empty}"
      sleep "${CHECK_INTERVAL_SECONDS}"
      continue
    fi

    log "gpu=${GPU_INDEX} free_mb=${free_mb} threshold_mb=${MIN_FREE_MB}"
    if (( free_mb > MIN_FREE_MB )); then
      launch_clsprec
      exit 0
    fi

    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

main "$@" >> "${WATCH_LOG}" 2>&1
