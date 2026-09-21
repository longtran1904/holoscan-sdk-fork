/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>

// Defined by compiling the unchanged reference main.cpp with a target-local main rename.
int run_original_ltsgemm_benchmark();

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct RunResult {
  int invocations = 0;
  int return_code = 1;
  bool completed = false;
  double entrypoint_wall_ms = 0;
  double app_run_wall_ms = 0;
  std::exception_ptr error;
};

void invoke_benchmark(RunResult& result) {
  ++result.invocations;
  if (result.invocations != 1) {
    result.completed = false;
    result.return_code = 1;
    return;
  }

  const auto start = Clock::now();
  try {
    result.return_code = run_original_ltsgemm_benchmark();
    result.entrypoint_wall_ms = elapsed_ms(start);
    result.completed = result.return_code == 0;
  } catch (...) {
    result.entrypoint_wall_ms = elapsed_ms(start);
    result.error = std::current_exception();
  }
}

class LtSgemmBenchmarkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LtSgemmBenchmarkOp)

  LtSgemmBenchmarkOp() = default;

  void set_result(std::shared_ptr<RunResult> result) { result_ = std::move(result); }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    invoke_benchmark(*result_);
  }

 private:
  std::shared_ptr<RunResult> result_;
};

class LtSgemmBenchmarkApp : public holoscan::Application {
 public:
  void set_result(std::shared_ptr<RunResult> result) { result_ = std::move(result); }

  void compose() override {
    auto benchmark =
        make_operator<LtSgemmBenchmarkOp>("ltsgemm", make_condition<holoscan::CountCondition>(1));
    benchmark->set_result(result_);
    add_operator(benchmark);
    scheduler(make_scheduler<holoscan::GreedyScheduler>("scheduler"));
  }

 private:
  std::shared_ptr<RunResult> result_;
};

void run_holoscan(const std::shared_ptr<RunResult>& result) {
  // Framework logs are disabled by default; the original benchmark keeps all of its output.
  holoscan::set_log_level(holoscan::LogLevel::OFF);
  auto app = holoscan::make_application<LtSgemmBenchmarkApp>();
  app->set_result(result);
  const auto start = Clock::now();
  try {
    app->run();
    result->app_run_wall_ms = elapsed_ms(start);
  } catch (...) {
    result->app_run_wall_ms = elapsed_ms(start);
    throw;
  }
}

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " [--runner=direct|holoscan] [--help]\n"
            << "Run the unchanged, complete LtSgemm sweep once (default: holoscan).\n"
            << "Holoscan hosts the sweep in one operator invocation; timings do not measure "
               "scheduler overhead per GEMM.\n";
}

void print_summary(const std::string& runner, const RunResult& result) {
  std::cout.imbue(std::locale::classic());
  std::cout << std::setprecision(17) << "LT_SGEMM_WRAPPER {\"schema_version\":1,\"runner\":\""
            << runner << "\",\"completed\":" << (result.completed ? "true" : "false")
            << ",\"invocations\":" << result.invocations
            << ",\"return_code\":" << result.return_code
            << ",\"entrypoint_wall_ms\":" << result.entrypoint_wall_ms;
  if (runner == "holoscan") {
    std::cout << ",\"app_run_wall_ms\":" << result.app_run_wall_ms;
  }
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string runner = "holoscan";
  bool runner_set = false;
  bool help = false;
  auto result = std::make_shared<RunResult>();

  // Validate the entire command line before constructing an application or touching CUDA.
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--help") {
      help = true;
    } else if (!runner_set && (argument == "--runner=direct" || argument == "--runner=holoscan")) {
      runner = argument.substr(std::string("--runner=").size());
      runner_set = true;
    } else {
      std::cerr << "Invalid argument: " << argument << '\n';
      print_usage(argv[0]);
      result->return_code = 2;
      print_summary(runner, *result);
      return result->return_code;
    }
  }
  if (help) {
    print_usage(argv[0]);
    return 0;
  }

  try {
    if (runner == "direct") {
      invoke_benchmark(*result);
    } else {
      run_holoscan(result);
    }
  } catch (...) {
    result->completed = false;
    result->error = std::current_exception();
  }

  result->completed = result->completed && result->invocations == 1 && !result->error;
  if (!result->completed && result->return_code == 0) {
    result->return_code = 1;
  }
  if (result->error) {
    try {
      std::rethrow_exception(result->error);
    } catch (const std::exception& error) {
      std::cerr << "LtSgemm run failed: " << error.what() << '\n';
    } catch (...) {
      std::cerr << "LtSgemm run failed with an unknown exception\n";
    }
  } else if (!result->completed) {
    std::cerr << "LtSgemm did not complete exactly one successful invocation\n";
  }
  print_summary(runner, *result);
  return result->return_code;
}
