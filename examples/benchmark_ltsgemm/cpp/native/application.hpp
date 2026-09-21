/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_APPLICATION_HPP
#define EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_APPLICATION_HPP

#include <memory>

#include "types.hpp"

namespace ltsgemm::native {

// Measures app->run() and retains failures, including exceptions caught by the executor.
void run_application(const RunOptions& options, const std::shared_ptr<RunResult>& result);

}  // namespace ltsgemm::native

#endif  // EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_APPLICATION_HPP
