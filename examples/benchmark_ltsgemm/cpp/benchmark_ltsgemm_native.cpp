/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include "native/application.hpp"
#include "native/support.hpp"
#include "native/types.hpp"

int main(int argc, char** argv) {
  using namespace ltsgemm::native;

  auto result = std::make_shared<RunResult>();
  const auto arguments = parse_arguments(argc, argv);
  if (arguments.outcome == CliOutcome::error) {
    print_argument_error(argv[0], arguments.invalid_argument);
    result->return_code = 2;
    print_summary(arguments.options, *result);
    return 2;
  }
  if (arguments.outcome == CliOutcome::help) {
    print_usage(argv[0]);
    return 0;
  }

  run_application(arguments.options, result);
  report_run_status(arguments.options, *result);
  return result->return_code;
}
