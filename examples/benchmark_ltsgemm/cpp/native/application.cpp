/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "application.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "execution.hpp"
#include "support.hpp"
#include "timing.hpp"

namespace ltsgemm::native {
namespace {

class NativeLtSgemmOp : public holoscan::Operator {
 public:
  NativeLtSgemmOp(RunOptions options, std::shared_ptr<RunResult> result,
                  std::shared_ptr<holoscan::CountCondition> condition)
      : Operator(std::move(condition)), options_(options), result_(std::move(result)) {}

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    ++result_->compute_calls;
    if (result_->error)
      return;
    try {
      if (result_->compute_calls == 1) {
        print_device_info(query_device_info());
      }
      if (!case_) {
        if (result_->completed_cases >= kCases)
          throw std::runtime_error("unexpected extra compute call");
        case_.emplace(
            CaseConfig{128 << (result_->completed_cases % 10), result_->completed_cases >= 10});
      }
      for (int i = 0; i < options_.gemms_per_tick; ++i) {
        case_->launch();
        ++result_->timed_gemms;
      }
      if (case_->complete()) {
        const auto statistics = case_->finish();
        const auto start = case_->start_time();
        const auto config = case_->config();
        case_.reset();  // Include host-vector destruction in case wall time.
        const double wall_ms = elapsed_ms(start);
        print_case_result({config, statistics, wall_ms});
        ++result_->completed_cases;
      }
    } catch (...) {
      result_->error = std::current_exception();
      case_.reset();
      // Preserve the exception even if the executor catches operator failures.
      throw;
    }
  }

 private:
  const RunOptions options_;
  const std::shared_ptr<RunResult> result_;
  // BenchmarkCase owns one heap-allocated state, just as the original case did.
  std::optional<BenchmarkCase> case_;
};

class NativeLtSgemmApp : public holoscan::Application {
 public:
  NativeLtSgemmApp(RunOptions options, std::shared_ptr<RunResult> result)
      : options_(options), result_(std::move(result)) {}

  void compose() override {
    auto op = make_operator<NativeLtSgemmOp>(
        "ltsgemm",
        options_,
        result_,
        make_condition<holoscan::CountCondition>(kCases * kRepeats / options_.gemms_per_tick));
    add_operator(op);
    scheduler(make_scheduler<holoscan::GreedyScheduler>("scheduler"));
  }

 private:
  const RunOptions options_;
  const std::shared_ptr<RunResult> result_;
};

}  // namespace

void run_application(const RunOptions& options, const std::shared_ptr<RunResult>& result) {
  try {
    holoscan::set_log_level(holoscan::LogLevel::OFF);
    auto app = holoscan::make_application<NativeLtSgemmApp>(options, result);
    const auto start = Clock::now();
    try {
      app->run();
    } catch (...) {
      result->error = std::current_exception();
    }
    result->app_run_wall_ms = elapsed_ms(start);
  } catch (...) {
    result->error = std::current_exception();
  }
}

}  // namespace ltsgemm::native
