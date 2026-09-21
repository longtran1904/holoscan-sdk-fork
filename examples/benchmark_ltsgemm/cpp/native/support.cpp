/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>

namespace ltsgemm::native {

CliResult parse_arguments(int argc, char** argv) {
  CliResult result;
  bool mode_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--help") {
      result.outcome = CliOutcome::help;
    } else if (!mode_set &&
               (argument == "--gemms-per-tick=1" || argument == "--gemms-per-tick=1000")) {
      result.options.gemms_per_tick = argument == "--gemms-per-tick=1" ? 1 : kRepeats;
      mode_set = true;
    } else {
      result.outcome = CliOutcome::error;
      result.invalid_argument = argument;
      return result;
    }
  }
  return result;
}

TimingStatistics calculate_statistics(std::vector<double> timings_us) {
  if (timings_us.empty()) {
    throw std::invalid_argument("cannot calculate statistics without timing samples");
  }
  double mean = 0, total = 0, squared_deviations = 0;
  for (size_t i = 0; i < timings_us.size(); ++i) {
    const double us = timings_us[i];
    total += us;
    const double delta = us - mean;
    mean += delta / (i + 1);
    squared_deviations += delta * (us - mean);
  }
  std::sort(timings_us.begin(), timings_us.end());
  const auto percentile = [&timings_us](double p) {
    const double index = p * (timings_us.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(index));
    const size_t upper = static_cast<size_t>(std::ceil(index));
    return static_cast<float>(timings_us[lower] +
                              (timings_us[upper] - timings_us[lower]) * (index - lower));
  };
  return {static_cast<float>(mean),
          static_cast<float>(std::sqrt(squared_deviations / timings_us.size())),
          static_cast<float>(timings_us.front()),
          percentile(0.50),
          percentile(0.99),
          total};
}

void print_device_info(const DeviceInfo& device) {
  std::printf("L2 cache size: %d bytes (%.2f MiB)\n",
              device.l2_cache_bytes,
              device.l2_cache_bytes / (1024.0 * 1024.0));
}

void print_case_result(const CaseResult& result) {
  const auto& statistics = result.statistics;
  std::printf(
      "GEMM (M=%d, N=128, K=128, L2 flush=%s): %.3f ± %.3f us, "
      "min=%.3f us, median=%.3f us, P99=%.3f us, CV=%.3f%%, %.3f TFLOP/s\n",
      result.config.m,
      result.config.flush_l2 ? "on" : "off",
      statistics.average_us,
      statistics.stddev_us,
      statistics.min_us,
      statistics.median_us,
      statistics.p99_us,
      100.0 * statistics.stddev_us / statistics.average_us,
      (2.0 * result.config.m * 128 * 128) / (statistics.average_us * 1e6));
  std::printf("  Timed GEMM total: %.3f ms, wall time: %.3f ms, GEMM/wall: %.3f%%\n",
              statistics.total_gpu_us / 1000,
              result.wall_ms,
              100.0 * statistics.total_gpu_us / (result.wall_ms * 1000));
}

void print_argument_error(const char* program, const std::string& argument) {
  std::cerr << "Invalid argument: " << argument << '\n';
  print_usage(program);
}

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " [--gemms-per-tick=1|1000] [--help]\n"
            << "Run the native 20-case LtSgemm sweep (default: 1 GEMM per tick).\n";
}

void print_summary(const RunOptions& options, const RunResult& result) {
  std::cout.imbue(std::locale::classic());
  std::cout << std::fixed << std::setprecision(6)
            << "LT_SGEMM_NATIVE {\"schema_version\":1,\"gemms_per_tick\":" << options.gemms_per_tick
            << ",\"completed_cases\":" << result.completed_cases
            << ",\"timed_gemms\":" << result.timed_gemms
            << ",\"compute_calls\":" << result.compute_calls
            << ",\"app_run_wall_ms\":" << result.app_run_wall_ms
            << ",\"completed\":" << (result.return_code == 0 ? "true" : "false")
            << ",\"return_code\":" << result.return_code << "}\n";
}

void report_run_status(const RunOptions& options, RunResult& result) {
  if (result.error) {
    try {
      std::rethrow_exception(result.error);
    } catch (const std::exception& error) {
      std::cerr << "Native LtSgemm run failed: " << error.what() << '\n';
    } catch (...) {
      std::cerr << "Native LtSgemm run failed with an unknown exception\n";
    }
  } else if (result.completed_cases == kCases && result.timed_gemms == kCases * kRepeats &&
             result.compute_calls == kCases * kRepeats / options.gemms_per_tick) {
    result.return_code = 0;
  } else {
    std::cerr << "Native LtSgemm did not complete the expected cases, GEMMs, and compute calls\n";
  }
  print_summary(options, result);
}

}  // namespace ltsgemm::native
