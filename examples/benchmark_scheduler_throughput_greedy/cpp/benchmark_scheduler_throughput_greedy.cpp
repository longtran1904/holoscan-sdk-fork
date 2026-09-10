/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <holoscan/holoscan.hpp>

// Increments a counter on each invocation.
class CountOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CountOp)

  CountOp() = default;

  void start() override { start_time_ = std::chrono::steady_clock::now(); }

  void stop() override { end_time_ = std::chrono::steady_clock::now(); }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    count_++;
  }

  int64_t count() const { return count_; }
  std::chrono::steady_clock::time_point start_time() const { return start_time_; }
  std::chrono::steady_clock::time_point end_time() const { return end_time_; }

 private:
  int64_t count_ = 0;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;
};

// Occupies the scheduler thread for at least 1 ms before incrementing the counter.
class BusyWaitCountOp : public CountOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(BusyWaitCountOp, CountOp)

  BusyWaitCountOp() = default;

  void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    while (std::chrono::steady_clock::now() < deadline) {
      // Poll the clock continuously: sleeping or yielding would release the CPU.
    }
    CountOp::compute(input, output, context);
  }
};

// Benchmarks GreedyScheduler throughput by counting operations over a period of time.
class BenchmarkSchedulerThroughputGreedyApp : public holoscan::Application {
 public:
  struct Options {
    int num_operators = 1;
    int num_operations = 100000;
    bool busy_wait = false;
  };

  struct Results {
    int64_t total_operations;
    float throughput_hz;
  };

  void set_options(const Options& options) { options_ = options; }

  void compose() override {
    for (int i = 0; i < options_.num_operators; i++) {
      const auto name = fmt::format("count_{}", i);
      auto condition = make_condition<holoscan::CountCondition>(options_.num_operations);
      if (options_.busy_wait) {
        count_ops_.push_back(make_operator<BusyWaitCountOp>(name, condition));
      } else {
        count_ops_.push_back(make_operator<CountOp>(name, condition));
      }
      add_operator(count_ops_.back());
    }

    scheduler(make_scheduler<holoscan::GreedyScheduler>("scheduler"));
  }

  Results results() const {
    int64_t total_operations = 0;
    std::chrono::steady_clock::time_point min_start_time = count_ops_[0]->start_time();
    std::chrono::steady_clock::time_point max_end_time = count_ops_[0]->end_time();
    for (const std::shared_ptr<CountOp>& count_op : count_ops_) {
      total_operations += count_op->count();
      min_start_time = std::min(min_start_time, count_op->start_time());
      max_end_time = std::max(max_end_time, count_op->end_time());
    }

    const float duration_s =
        std::chrono::duration_cast<std::chrono::duration<float>>(max_end_time - min_start_time)
            .count();
    return {total_operations, total_operations / duration_s};
  }

 private:
  Options options_;
  std::vector<std::shared_ptr<CountOp>> count_ops_;
};

namespace {

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [--busy_wait] [--help]" << '\n';
  std::cout << "  --busy_wait: Busy-wait 1 ms before each increment; run 1,000 operations"
               " per operator.\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool busy_wait = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--busy_wait") {
      busy_wait = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    std::cerr << "Unknown argument: " << arg << '\n';
    print_usage(argv[0]);
    return 1;
  }

  const std::vector<int> operator_counts = {1, 2, 4, 8, 12, 14, 16};
  std::vector<BenchmarkSchedulerThroughputGreedyApp::Results> trial_results;

  for (const int num_operators : operator_counts) {
    BenchmarkSchedulerThroughputGreedyApp::Options options;
    options.num_operators = num_operators;
    options.busy_wait = busy_wait;
    options.num_operations = busy_wait ? 1000 : 10000000;

    auto app = holoscan::make_application<BenchmarkSchedulerThroughputGreedyApp>();
    app->set_options(options);
    app->run();
    trial_results.push_back(app->results());
  }

  std::cout << "\nGreedyScheduler Throughput Benchmark Results:" << '\n';
  std::cout << "Operator: " << (busy_wait ? "BusyWaitCountOp (1 ms busy wait)" : "CountOp")
            << '\n';
  std::cout << fmt::format("\n| {:>5} | {:>9} | {:>10} | {:>12} |",
                           "Trial", "Operators", "Total Ops", "Ops/s")
            << '\n';
  std::cout << "|-------|-----------|------------|--------------|" << '\n';

  for (size_t i = 0; i < operator_counts.size(); i++) {
    const auto& results = trial_results[i];
    std::cout << fmt::format("| {:>5} | {:>9} | {:>10} | {:>12.1f} |",
                             i,
                             operator_counts[i],
                             results.total_operations,
                             results.throughput_hz)
              << '\n';
  }

  return 0;
}
