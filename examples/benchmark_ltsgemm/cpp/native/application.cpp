/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "application.hpp"

#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "execution.hpp"
#include "support.hpp"
#include "timing.hpp"

namespace ltsgemm::native {
namespace {

class NativeLtSgemmOp : public holoscan::Operator {
 public:
  NativeLtSgemmOp(RunOptions options, std::shared_ptr<RunResult> result, int operator_index, int m,
                  bool flush_l2, std::shared_ptr<holoscan::CountCondition> condition)
      : Operator(std::move(condition)),
        options_(options),
        result_(std::move(result)),
        operator_index_(operator_index),
        m_(m),
        flush_l2_(flush_l2) {}

  void initialize() override { Operator::initialize(); }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    bool first_call;
    {
      std::lock_guard<std::mutex> lock(result_->counter_mutex);
      first_call = ++result_->compute_calls == 1;
    }
    if (result_->error)
      return;
    try {
      if (first_call) {
        print_device_info(query_device_info());
      }
      if (!case_) {
        case_.emplace(m_, flush_l2_);
      }
      for (int i = 0; i < options_.gemms_per_tick; ++i) {
        case_->launch();
        {
          std::lock_guard<std::mutex> lock(result_->counter_mutex);
          ++result_->timed_gemms;
        }
      }
      if (case_->complete()) {
        print_case_result(case_->finish(), operator_index_, options_.operator_count);
        {
          std::lock_guard<std::mutex> lock(result_->counter_mutex);
          ++result_->completed_cases;
        }
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
  const int operator_index_;
  const int m_;
  const bool flush_l2_;
  std::optional<BenchmarkCase> case_;
};

class NativeLtSgemmApp : public holoscan::Application {
 public:
  NativeLtSgemmApp(RunOptions options, std::shared_ptr<RunResult> result, int m, bool flush_l2)
      : options_(options), result_(std::move(result)), m_(m), flush_l2_(flush_l2) {}

  void compose() override {
    for (int i = 0; i < options_.operator_count; ++i) {
      auto op = make_operator<NativeLtSgemmOp>(
          "ltsgemm_" + std::to_string(i),
          options_,
          result_,
          i,
          m_,
          flush_l2_,
          make_condition<holoscan::CountCondition>(kRepeats / options_.gemms_per_tick));
      add_operator(op);
    }
    // GreedyScheduler
    // scheduler(make_scheduler<holoscan::GreedyScheduler>("scheduler"));

    // // Event-Based Scheduler for better parallelism
    scheduler(make_scheduler<holoscan::EventBasedScheduler>(
        "ebs",
        holoscan::Arg("worker_thread_number", 13),
        holoscan::Arg("enable_queue_stealing", false),
        holoscan::Arg("enable_worker_postcheck_fastpath", false),
        holoscan::Arg("internal_event_shard_count", static_cast<int64_t>(1)),
        holoscan::Arg("dispatcher_internal_pop_batch_size", static_cast<int64_t>(1)),
        holoscan::Arg("wait_state_shard_count", static_cast<int64_t>(1)),
        holoscan::Arg("log_perf_stats", true)));

    printf(
        "Running Event-Based Scheduler with %d threads, %d operators, %d GEMMs per tick, m=%d, "
        "flush_l2=%s\n",
        13,
        options_.operator_count,
        options_.gemms_per_tick,
        m_,
        flush_l2_ ? "on" : "off");
  }

 private:
  const RunOptions options_;
  const std::shared_ptr<RunResult> result_;
  const int m_;
  const bool flush_l2_;
};

}  // namespace

void run_application(const RunOptions& options, const std::shared_ptr<RunResult>& result) {
  // Run application for each case
  // try {
  //   holoscan::set_log_level(holoscan::LogLevel::OFF);
  //   for (int case_index = 0; case_index < kCases; ++case_index) {
  //     const int m = 128 << (case_index % 10);
  //     const bool flush_l2 = case_index >= 10;
  //     auto app = holoscan::make_application<NativeLtSgemmApp>(options, result, m, flush_l2);
  //     const auto start = Clock::now();
  //     try {
  //       app->run();
  //     } catch (...) {
  //       result->error = std::current_exception();
  //     }
  //     result->app_run_wall_ms += elapsed_ms(start);
  //     if (result->error)
  //       break;
  //   }
  // } catch (...) {
  //   result->error = std::current_exception();
  // }

  // Run application for only 1 case m=65536, flush_l2=true
  try {
    holoscan::set_log_level(holoscan::LogLevel::OFF);
    const int m = 65536;
    const bool flush_l2 = true;
    auto app = holoscan::make_application<NativeLtSgemmApp>(options, result, m, flush_l2);
    const auto start = Clock::now();
    try {
      app->run();
    } catch (...) {
      result->error = std::current_exception();
    }
    result->app_run_wall_ms += elapsed_ms(start);
  } catch (...) {
    result->error = std::current_exception();
  }
}

}  // namespace ltsgemm::native
