/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "support.hpp"

namespace {

void expect_close(double actual, double expected, const char* name) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5) {
    throw std::runtime_error(std::string(name) + ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

void test_equal_samples() {
  const auto result = ltsgemm::native::calculate_statistics(std::vector<double>(1000, 2.5));
  expect_close(result.average_us, 2.5, "equal mean");
  expect_close(result.stddev_us, 0, "equal population standard deviation");
  expect_close(result.min_us, 2.5, "equal minimum");
  expect_close(result.median_us, 2.5, "equal median");
  expect_close(result.p99_us, 2.5, "equal P99");
  expect_close(result.total_gpu_us, 2500, "equal total");
}

void test_interpolated_percentiles() {
  const std::vector<double> samples{9, 1, 5, 3};
  const auto result = ltsgemm::native::calculate_statistics(samples);
  expect_close(result.average_us, 4.5, "mean");
  expect_close(result.stddev_us, std::sqrt(8.75), "population standard deviation");
  expect_close(result.min_us, 1, "minimum");
  expect_close(result.median_us, 4, "interpolated median");
  expect_close(result.p99_us, 8.88, "interpolated P99");
  expect_close(result.total_gpu_us, 18, "total");
  if (samples != std::vector<double>{9, 1, 5, 3}) {
    throw std::runtime_error("statistics changed the caller's samples");
  }
}

void test_single_sample() {
  const auto result = ltsgemm::native::calculate_statistics({1.25});
  expect_close(result.average_us, 1.25, "single mean");
  expect_close(result.stddev_us, 0, "single population standard deviation");
  expect_close(result.min_us, 1.25, "single minimum");
  expect_close(result.median_us, 1.25, "single median");
  expect_close(result.p99_us, 1.25, "single P99");
  expect_close(result.total_gpu_us, 1.25, "single total");
}

void test_empty_samples() {
  try {
    ltsgemm::native::calculate_statistics({});
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("empty samples were accepted");
}

}  // namespace

int main() {
  try {
    test_equal_samples();
    test_interpolated_percentiles();
    test_single_sample();
    test_empty_samples();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
