#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
CONDA_ENV="${EASYR1_CONDA_ENV:-easyr1}"
RAY_BIN="${CONDA_ENV}/bin/ray"
HEAD_HOST="${EASYR1_RAY_HEAD_HOST:-$(hostname -f)}"
HEAD_IP="${EASYR1_RAY_HEAD_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
HEAD_PORT="${EASYR1_RAY_HEAD_PORT:-6379}"
HEAD_CPUS="${EASYR1_RAY_HEAD_CPUS:-32}"
HEAD_GPUS="${EASYR1_RAY_HEAD_GPUS:-0}"
WORKER_CPUS="${EASYR1_RAY_WORKER_CPUS:-64}"
WORKER_GPUS="${EASYR1_RAY_WORKER_GPUS:-8}"
SSH_KNOWN_HOSTS="${EASYR1_RAY_KNOWN_HOSTS:-/tmp/easyr1_known_hosts}"
STOP_TIMEOUT="${EASYR1_RAY_STOP_TIMEOUT:-30}"
RAY_CLEAN_PATTERN='[r]ay start|[r]aylet|[g]cs_server|[r]ay-dashboard|[d]ashboard|[m]onitor.py|[l]og_monitor.py|[r]untime_env_agent|[p]lasma_store'

DEFAULT_WORKERS=(
  # 替换为你的 Ray worker 节点 hostname；或导出环境变量 EASYR1_RAY_WORKERS="host1 host2 ..." 覆盖
  worker-0
  worker-1
  worker-2
  worker-3
  worker-4
  worker-5
)

if [[ -n "${EASYR1_RAY_WORKERS:-}" ]]; then
  read -r -a WORKERS <<< "${EASYR1_RAY_WORKERS}"
else
  WORKERS=("${DEFAULT_WORKERS[@]}")
fi

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile="${SSH_KNOWN_HOSTS}"
)

run_local() {
  PATH="${CONDA_ENV}/bin:${PATH}" "$@"
}

run_remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${host}" "$*"
}

stop_node() {
  local host="$1"
  local stop_cmd="timeout '${STOP_TIMEOUT}' '${RAY_BIN}' stop --force >/tmp/easyr1_ray_stop.log 2>&1 || true; pkill -9 -f '${RAY_CLEAN_PATTERN}' >/tmp/easyr1_ray_pkill.log 2>&1 || true"
  if [[ "${host}" == "$(hostname -f)" || "${host}" == "$(hostname)" ]]; then
    timeout "${STOP_TIMEOUT}" "${RAY_BIN}" stop --force >/tmp/easyr1_ray_stop.${host}.log 2>&1 || true
    pkill -9 -f "${RAY_CLEAN_PATTERN}" >/tmp/easyr1_ray_pkill.${host}.log 2>&1 || true
  else
    run_remote "${host}" "${stop_cmd}" || true
  fi
}

start_head() {
  local log_file="/tmp/easyr1_ray_head.nohup.log"
  echo "Starting Ray head on ${HEAD_HOST} (${HEAD_IP}:${HEAD_PORT}), GPUs=${HEAD_GPUS}"
  if [[ "${HEAD_HOST}" == "$(hostname -f)" || "${HEAD_HOST}" == "$(hostname)" ]]; then
    PATH="${CONDA_ENV}/bin:${PATH}" setsid -f "${RAY_BIN}" start --head \
      --node-ip-address="${HEAD_IP}" \
      --port="${HEAD_PORT}" \
      --include-dashboard=false \
      --disable-usage-stats \
      --num-cpus="${HEAD_CPUS}" \
      --num-gpus="${HEAD_GPUS}" \
      --block >"${log_file}" 2>&1 < /dev/null &
    sleep 5
    if ! pgrep -f "[r]ay start --head" >/dev/null; then
      tail -n 120 "${log_file}" >&2 || true
      return 1
    fi
  else
    run_remote "${HEAD_HOST}" "PATH='${CONDA_ENV}/bin':\$PATH setsid -f '${RAY_BIN}' start --head --node-ip-address='${HEAD_IP}' --port='${HEAD_PORT}' --include-dashboard=false --disable-usage-stats --num-cpus='${HEAD_CPUS}' --num-gpus='${HEAD_GPUS}' --block >/tmp/easyr1_ray_head.nohup.log 2>&1 < /dev/null; sleep 5; pgrep -f '[r]ay start --head' >/dev/null"
  fi
  echo "Ray head log: ${HEAD_HOST}:${log_file}"
}

start_worker() {
  local host="$1"
  echo "Starting Ray worker on ${host}, joining ${HEAD_IP}:${HEAD_PORT}, GPUs=${WORKER_GPUS}"
  run_remote "${host}" "PATH='${CONDA_ENV}/bin':\$PATH setsid -f '${RAY_BIN}' start --address='${HEAD_IP}:${HEAD_PORT}' --num-cpus='${WORKER_CPUS}' --num-gpus='${WORKER_GPUS}' --disable-usage-stats --block >/tmp/easyr1_ray_worker.nohup.log 2>&1 < /dev/null; sleep 5; pgrep -f '[r]ay start --address' >/dev/null"
  echo "Ray worker log: ${host}:/tmp/easyr1_ray_worker.nohup.log"
}

status_cluster() {
  RAY_ADDRESS="${HEAD_IP}:${HEAD_PORT}" run_local "${RAY_BIN}" status
}

case "${ACTION}" in
  start)
    stop_node "${HEAD_HOST}"
    for host in "${WORKERS[@]}"; do
      stop_node "${host}"
    done
    start_head
    sleep 5
    for host in "${WORKERS[@]}"; do
      start_worker "${host}"
    done
    sleep 8
    status_cluster
    ;;
  stop)
    for host in "${WORKERS[@]}"; do
      stop_node "${host}"
    done
    stop_node "${HEAD_HOST}"
    ;;
  status)
    status_cluster
    ;;
  *)
    echo "Usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
