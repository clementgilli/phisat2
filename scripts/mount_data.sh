#!/usr/bin/env bash
set -euo pipefail

BUCKET="${PHISATNET_BUCKET:-ESA-philab/PhiSatNet}"
MOUNT_DIR="${PHISATNET_MOUNT_DIR:-${PWD}/data/PhiSatNet}"
CACHE_DIR="${PHISATNET_CACHE_DIR:-${HOME}/.cache/hf-mount/phisatnet}"
HF_BIN="${HF_BIN:-hf}"

usage() {
  cat <<'USAGE'
Mount or sync the Hugging Face bucket ESA-philab/PhiSatNet locally.

When hf-mount is available, this script mounts the bucket as a filesystem.
Otherwise, it falls back to Hugging Face CLI sync and materializes files under
the same local path.

Usage:
  scripts/mount_phisatnet_bucket.sh [mount|status|umount|help]

Environment:
  PHISATNET_MOUNT_DIR   Local mount path. Default: ./data/PhiSatNet
  PHISATNET_CACHE_DIR   hf-mount cache path. Default: ~/.cache/hf-mount/phisatnet
  PHISATNET_BUCKET      Bucket id. Default: ESA-philab/PhiSatNet
  HF_TOKEN              Hugging Face token, required if the bucket is private/gated.
  HF_BIN                Hugging Face CLI executable. Default: hf
  PHISATNET_READ_WRITE  Set to 1 to mount read-write. Default is read-only.
  HF_MOUNT_EXTRA_ARGS   Extra arguments passed through to hf-mount start.
  HF_SYNC_EXTRA_ARGS    Extra arguments passed through to hf buckets sync.

Examples:
  scripts/mount_phisatnet_bucket.sh
  PHISATNET_MOUNT_DIR=/mnt/phisatnet scripts/mount_phisatnet_bucket.sh mount
  scripts/mount_phisatnet_bucket.sh status
  scripts/mount_phisatnet_bucket.sh umount
USAGE
}

require_hf_cli() {
  if command -v "${HF_BIN}" >/dev/null 2>&1; then
    return
  fi

  cat >&2 <<ERROR
Hugging Face CLI '${HF_BIN}' is required but was not found on PATH.

Install project dependencies with:
  uv sync

Then run:
  uv run scripts/mount_phisatnet_bucket.sh
ERROR
  exit 127
}

is_mounted() {
  mount | awk -v mount_dir="${MOUNT_DIR}" '$3 == mount_dir { found = 1 } END { exit !found }'
}

mount_bucket() {
  mkdir -p "${MOUNT_DIR}" "${CACHE_DIR}"

  if ! command -v hf-mount >/dev/null 2>&1; then
    sync_bucket
    return
  fi

  if is_mounted; then
    echo "Already mounted: ${MOUNT_DIR}"
    return
  fi

  args=(start --cache-dir "${CACHE_DIR}")

  if [[ "${PHISATNET_READ_WRITE:-0}" != "1" ]]; then
    args+=(--read-only)
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    args+=(--hf-token "${HF_TOKEN}")
  fi

  if [[ -n "${HF_MOUNT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args+=(${HF_MOUNT_EXTRA_ARGS})
  fi

  args+=(bucket "${BUCKET}" "${MOUNT_DIR}")

  echo "Mounting hf://buckets/${BUCKET} at ${MOUNT_DIR}"
  hf-mount "${args[@]}"
  echo "Mounted. Stop it with: scripts/mount_phisatnet_bucket.sh umount"
}

sync_bucket() {
  require_hf_cli
  mkdir -p "${MOUNT_DIR}"

  args=(buckets sync "hf://buckets/${BUCKET}" "${MOUNT_DIR}")

  if [[ -n "${HF_TOKEN:-}" ]]; then
    args+=(--token "${HF_TOKEN}")
  fi

  if [[ -n "${HF_SYNC_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args+=(${HF_SYNC_EXTRA_ARGS})
  fi

  echo "hf-mount not found; syncing hf://buckets/${BUCKET} to ${MOUNT_DIR}"
  "${HF_BIN}" "${args[@]}"
  echo "Synced ${MOUNT_DIR}"
}

unmount_bucket() {
  if ! command -v hf-mount >/dev/null 2>&1; then
    echo "hf-mount is not installed; sync fallback has no mounted filesystem to stop."
    return
  fi

  hf-mount stop "${MOUNT_DIR}"
}

case "${1:-mount}" in
  mount)
    mount_bucket
    ;;
  status)
    if command -v hf-mount >/dev/null 2>&1; then
      hf-mount status
    else
      require_hf_cli
      "${HF_BIN}" buckets info "${BUCKET}"
    fi
    ;;
  umount|unmount|stop)
    unmount_bucket
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
