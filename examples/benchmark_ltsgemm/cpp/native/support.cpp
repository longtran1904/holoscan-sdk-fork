/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "support.hpp"

#include <algorithm>
#include <charconv>
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
  bool operator_count_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--help") {
      result.outcome = CliOutcome::help;
    } else if (!mode_set &&
               (argument == "--gemms-per-tick=1" || argument == "--gemms-per-tick=1000")) {
      result.options.gemms_per_tick = argument == "--gemms-per-tick=1" ? 1 : kRepeats;
      mode_set = true;
    } else if (!operator_count_set && argument.rfind("--operators=", 0) == 0) {
      const auto value = argument.substr(sizeof("--operators=") - 1);
      int operator_count = 0;
      const auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), operator_count);
      if (value.empty() || parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() ||
          operator_count < 1) {
        result.outcome = CliOutcome::error;
        result.invalid_argument = argument;
        return result;
      }
      result.options.operator_count = operator_count;
      operator_count_set = true;
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

void print_case_result(const CaseResult& result, int operator_index, int operator_count) {
  const auto& statistics = result.statistics;
  std::printf("Operator %d/%d\n", operator_index + 1, operator_count);
  std::printf(
      "GEMM (M=%d, N=128, K=128, L2 flush=%s): %.3f ± %.3f us, "
      "min=%.3f us, median=%.3f us, P99=%.3f us, CV=%.3f%%, %.3f TFLOP/s\n",
      result.m,
      result.flush_l2 ? "on" : "off",
      statistics.average_us,
      statistics.stddev_us,
      statistics.min_us,
      statistics.median_us,
      statistics.p99_us,
      100.0 * statistics.stddev_us / statistics.average_us,
      (2.0 * result.m * 128 * 128) / (statistics.average_us * 1e6));
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
  std::cout << "Usage: " << program << " [--gemms-per-tick=1|1000] [--operators=N] [--help]\n"
            << "Run N identical LtSgemm operators (defaults: 1 operator, 1 GEMM per tick).\n";
}

void print_summary(const RunOptions& options, const RunResult& result) {
  std::cout.imbue(std::locale::classic());
  std::cout << std::fixed << std::setprecision(6)
            << "LT_SGEMM_NATIVE {\"schema_version\":2,\"gemms_per_tick\":" << options.gemms_per_tick
            << ",\"operator_count\":" << options.operator_count
            << ",\"completed_cases\":" << result.completed_cases
            << ",\"timed_gemms\":" << result.timed_gemms
            << ",\"compute_calls\":" << result.compute_calls
            << ",\"app_run_wall_ms\":" << result.app_run_wall_ms
            << ",\"completed\":" << (result.return_code == 0 ? "true" : "false")
            << ",\"return_code\":" << result.return_code << "}\n";
}

void report_run_status(const RunOptions& options, RunResult& result) {
  const int expected_cases = kCases * options.operator_count;
  if (result.error) {
    try {
      std::rethrow_exception(result.error);
    } catch (const std::exception& error) {
      std::cerr << "Native LtSgemm run failed: " << error.what() << '\n';
    } catch (...) {
      std::cerr << "Native LtSgemm run failed with an unknown exception\n";
    }
  } else if (result.completed_cases == expected_cases &&
             result.timed_gemms == expected_cases * kRepeats &&
             result.compute_calls == expected_cases * kRepeats / options.gemms_per_tick) {
    result.return_code = 0;
  } else {
    std::cerr << "Native LtSgemm did not complete the expected cases, GEMMs, and compute calls\n";
  }
  print_summary(options, result);
}

}  // namespace ltsgemm::native
