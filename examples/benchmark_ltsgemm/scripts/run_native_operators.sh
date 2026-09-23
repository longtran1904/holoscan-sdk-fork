#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example_dir="$(cd "${script_dir}/.." && pwd)"
binary="${LTSGEMM_NATIVE_BINARY:-${example_dir}/build-ltsgemm-holoscan/cpp/benchmark_ltsgemm_native}"
operators=("1" "2" "4" "8" "16" "32" "64")

# operators=("10" "20" "30")

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  echo "Usage: $0 [--help]"
  echo "Configured operators: ${operators[*]}"
  echo "Set LTSGEMM_NATIVE_BINARY to use a different executable."
  exit 0
fi

if [[ ! -x "$binary" ]]; then
  echo "Native benchmark is not executable: $binary" >&2
  echo "Build it first with ${script_dir}/build_native.sh" >&2
  exit 1
fi

for count in "${operators[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid operator count: $count (expected a positive integer)" >&2
    exit 2
  fi
done

results_dir="${example_dir}/results"
run_dir="${results_dir}/app-run-$(date +%Y-%m-%d_%H-%M-%S_%N)"
mkdir -p "$results_dir"
mkdir "$run_dir"
echo "Results: $run_dir"

for count in "${operators[@]}"; do
  file="run-operators-${count}"
  echo "Running --operators=${count}"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
     HOLOSCAN_LOG_LEVEL="${HOLOSCAN_LOG_LEVEL:-OFF}" \
     /usr/bin/time -f 'process_wall_seconds=%e\nmax_rss_kb=%M\nexit_status=%x' \
       -o "${run_dir}/${file}.time.txt" \
       "$binary" "--operators=${count}" \
       > "${run_dir}/${file}.stdout.log" \
       2> "${run_dir}/${file}.stderr.log"; then
    echo "Completed --operators=${count}"
  else
    status=$?
    echo "Run failed for --operators=${count} (exit ${status}); see ${run_dir}/${file}.stderr.log" >&2
    exit "$status"
  fi
done

# AI-generated script to summarize results into CSV file
plot_python="${example_dir}/benchmark-env/bin/python3"
python3 "${script_dir}/summarize_native_operators.py" "$run_dir"
"$plot_python" "${script_dir}/plot_native_operator_latencies.py" "$run_dir"

if ! command -v codex >/dev/null 2>&1; then
  echo "Codex CLI is not available; CSV is saved in $run_dir" >&2
  exit 1
fi

# AI-session to summarize the raw results into a Markdown reports
# report="${run_dir}/COMPARISON.md"
# template="${results_dir}/COMPARISON.md"
# echo "Generating experiment report: $report"
# if codex exec --sandbox read-only --cd "$example_dir" \
#     --output-last-message "$report" \
#     "Read the experiment results in $run_dir, especially comparison.csv and the raw logs. Use $template only as a section and style reference, not as a source of measurements. Write a complete Markdown report with Result, Comparison, Run configuration, and Artifacts sections. Use only this run's operator counts and measurements; calculate comparisons from its CSV and state limitations for a single-trial sweep. Peak RSS is system RAM, not GPU memory, and summed GPU-event time divided by app time is not a concurrency or utilization measurement. Do not rerun benchmarks or edit repository files. Your final response must be exactly the report Markdown, with no preamble or code fence."; then
#   if [[ ! -s "$report" ]]; then
#     echo "Codex returned without writing a report: $report" >&2
#     exit 1
#   fi
# else
#   status=$?
#   echo "Codex report generation failed (exit ${status}); CSV is saved in $run_dir" >&2
#   exit "$status"
# fi

echo "All runs completed. Results: $run_dir"
