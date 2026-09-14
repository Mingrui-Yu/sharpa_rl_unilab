#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <scale1> [scale2 ...] (available: 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5)"
  echo "Environment:"
  echo "  SHARPA_GRASP_CACHE_PATH=<prefix>     optional recorder output prefix"
  echo "  SHARPA_GRASP_TARGET=<count>          optional cache size"
  echo "  SHARPA_GRASP_NUM_ENVS=<count>        optional vector-env count"
  exit 1
fi

extra_args=(training.no_play=true)
if [ -n "${SHARPA_GRASP_CACHE_PATH:-}" ]; then
  extra_args+=("env.recorders.grasp_cache.params.output_prefix=${SHARPA_GRASP_CACHE_PATH}")
fi
if [ -n "${SHARPA_GRASP_TARGET:-}" ]; then
  extra_args+=("env.recorders.grasp_cache.params.target=${SHARPA_GRASP_TARGET}")
fi
if [ -n "${SHARPA_GRASP_NUM_ENVS:-}" ]; then
  extra_args+=("algo.num_envs=${SHARPA_GRASP_NUM_ENVS}")
fi

for scale in "$@"; do
  case "${scale}" in
    0.8|0.9|1|1.1|1.2|1.3|1.4|1.5) ;;
    *) echo "Unsupported fixed object scale ${scale}"; exit 1 ;;
  esac
  source_file="src/sharpa_rl_unilab/assets/robots/sharpa_wave/scene_scale_${scale}.xml"
  if [ ! -f "${source_file}" ]; then
    echo "Missing fixed variant XML: ${source_file}"
    exit 1
  fi
  echo "[sharpa_collect_grasps] collecting backend=mujoco scale=${scale}"
  SHARPA_GRASP_SCALE="${scale}" SHARPA_GRASP_VARIANT_FILE="${source_file}" \
    uv run sharpa-train --algo ppo --sim mujoco --task sharpa_inhand_grasp \
      "${extra_args[@]}"
done
