/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_SUPPORT_HPP
#define EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_SUPPORT_HPP

#include <string>
#include <vector>

#include "types.hpp"

namespace ltsgemm::native {

enum class CliOutcome { run, help, error };

struct CliResult {
  RunOptions options;
  CliOutcome outcome = CliOutcome::run;
  std::string invalid_argument;
};

CliResult parse_arguments(int argc, char** argv);

// Population standard deviation and linear interpolation at p * (sample count - 1).
// Takes ownership so callers can move the event samples without another allocation.
// Throws std::invalid_argument for an empty sample set.
TimingStatistics calculate_statistics(std::vector<double> timings_us);

void print_device_info(const DeviceInfo& device);
void print_case_result(const CaseResult& result, int operator_index, int operator_count);
void print_usage(const char* program);
void print_argument_error(const char* program, const std::string& argument);
void print_summary(const RunOptions& options, const RunResult& result);
void report_run_status(const RunOptions& options, RunResult& result);

}  // namespace ltsgemm::native

#endif  // EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_SUPPORT_HPP
