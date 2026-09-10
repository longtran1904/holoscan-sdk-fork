/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <holoscan/holoscan.hpp>

// Increments a counter on each invocation.
class CountOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CountOp)

  CountOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.param(index, "index", "Index", "Index of this operator");
  }

  void set_progress_log(std::shared_ptr<std::ofstream> stream) {
    progress_log = std::move(stream);
  }

  void start() override {
    // Use wall clock instead of Holoscan's internal clock in order to
    // separately measurement from implementation.
    start_time_ = std::chrono::steady_clock::now();
  }

  void stop() override {
    end_time_ = std::chrono::steady_clock::now();

    // Write timestamp checkpoints to file for plotting.
    for (const auto& timestamp : timestamps_) {
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.second - start_time_);
      (*progress_log) << timestamp.first << ',' << elapsed.count() << '\n';
    }
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    count_++;
    if (count_ % 10000 == 0) {
      timestamps_.push_back(make_pair(index.get(), std::chrono::steady_clock::now()));
    }
  };

  int64_t count() const { return count_; }
  std::chrono::steady_clock::time_point start_time() const { return start_time_; }
  std::chrono::steady_clock::time_point end_time() const { return end_time_; }
  std::vector<std::pair<int64_t, std::chrono::steady_clock::time_point>> timestamps() const {
    return timestamps_;
  }

 private:
  // Operator Parameters
  holoscan::Parameter<int64_t> index;

  // Internal states
  std::shared_ptr<std::ofstream> progress_log;
  int64_t count_ = 0;
  std::vector<std::pair<int64_t, std::chrono::steady_clock::time_point>> timestamps_;
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

// Benchmarks scheduler throughput by counting operations over a period of time.
class BenchmarkSchedulerThroughputApp : public holoscan::Application {
 public:
  struct Options {
    // Number of worker threads used by the scheduler.
    int num_threads = 1;

    // Number of operators.
    int num_operators = 1;

    // Operations to execute per operator.
    int num_operations = 100000;

    bool busy_wait = false;

    // Whether EventBasedScheduler workers may steal ready jobs from other worker queues.
    bool enable_queue_stealing = false;

    // Whether EventBasedScheduler workers use the postcheck fastpath after executeEntity().
    bool enable_postcheck_fastpath = false;
  };

  struct Results {
    int total_operations;
    float throughput_hz;
  };

  void set_options(const Options& options) { options_ = options; }

  void compose() override {
    // Create a log file to record timestamp checkpoints for plotting.
    std::string progress_log_filename = fmt::format("progress_log_{}ops_{}threads_{}operators.txt",
                                                    options_.num_operations,
                                                    options_.num_threads,
                                                    options_.num_operators);
    auto progress_log = std::make_shared<std::ofstream>(progress_log_filename);
    if (!progress_log->is_open()) {
      throw std::runtime_error("Failed to open progress.csv");
    }

    for (int i = 0; i < options_.num_operators; i++) {
      const auto name = fmt::format("count_{}", i);
      auto condition = make_condition<holoscan::CountCondition>(options_.num_operations);
      if (options_.busy_wait) {
        count_ops_.push_back(
            make_operator<BusyWaitCountOp>(name, condition, holoscan::Arg("index", i)));
      } else {
        count_ops_.push_back(make_operator<CountOp>(name, condition, holoscan::Arg("index", i)));
      }
      count_ops_.back()->set_progress_log(progress_log);
      add_operator(count_ops_.back());
    }

    scheduler(make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler",
        holoscan::Arg("worker_thread_number", static_cast<int64_t>(options_.num_threads)),
        holoscan::Arg("enable_queue_stealing", options_.enable_queue_stealing),
        holoscan::Arg("enable_worker_postcheck_fastpath", options_.enable_postcheck_fastpath)));
  }

  Results results() {
    // Aggregate statistics from all operator flows.
    int total_operations = 0;
    std::chrono::steady_clock::time_point min_start_time = count_ops_[0]->start_time();
    std::chrono::steady_clock::time_point max_end_time = count_ops_[0]->end_time();
    for (const std::shared_ptr<CountOp>& count_op : count_ops_) {
      total_operations += count_op->count();
      min_start_time = std::min(min_start_time, count_op->start_time());
      max_end_time = std::max(max_end_time, count_op->end_time());
    }

    // Calculate and return the results.
    float duration_s =
        std::chrono::duration_cast<std::chrono::duration<float>>(max_end_time - min_start_time)
            .count();
    Results results;
    results.total_operations = total_operations;
    results.throughput_hz = total_operations / duration_s;
    return results;
  }

 private:
  Options options_;

  std::vector<std::shared_ptr<CountOp>> count_ops_;
};

namespace {

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name
            << " [--enable_queue_stealing] [--enable_postcheck_fastpath] [--busy_wait] [--help]"
            << '\n';
  std::cout << "  --busy_wait: Busy-wait at least 1 ms before each increment; run 1,000"
               " operations per operator (default: 100,000 without waiting).\n";
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkSchedulerThroughputApp::Options scheduler_options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--enable_queue_stealing") {
      scheduler_options.enable_queue_stealing = true;
    } else if (arg == "--enable_postcheck_fastpath") {
      scheduler_options.enable_postcheck_fastpath = true;
    } else if (arg == "--busy_wait") {
      scheduler_options.busy_wait = true;
      scheduler_options.num_operations = 1000;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }

  auto make_trial_options = [&](int num_threads,
                                int num_operators) -> BenchmarkSchedulerThroughputApp::Options {
    BenchmarkSchedulerThroughputApp::Options options = scheduler_options;
    options.num_threads = num_threads;
    options.num_operators = num_operators;
    return options;
  };

  // Construct trial options to benchmark: {num_threads, num_operators}.
  // Default:
  //   std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = {
  //       make_trial_options(1, 1),
  //       make_trial_options(2, 2),
  //       make_trial_options(4, 4),
  //       make_trial_options(8, 8),
  //       make_trial_options(16, 16),
  //       make_trial_options(2, 8),
  //   };

  std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = {
      make_trial_options(1, 16),
      make_trial_options(2, 16),
      make_trial_options(4, 16),
      make_trial_options(8, 16),
      make_trial_options(10, 16),
      make_trial_options(12, 16),
      make_trial_options(13, 16),
      make_trial_options(14, 16),
      make_trial_options(16, 16),
  };
  std::vector<BenchmarkSchedulerThroughputApp::Results> trial_results;

  for (BenchmarkSchedulerThroughputApp::Options& options : trial_options) {
    auto app = holoscan::make_application<BenchmarkSchedulerThroughputApp>();
    app->set_options(options);
    app->run();
    trial_results.push_back(app->results());
  }

  std::cout << "\nScheduler Throughput Benchmark Results:" << '\n';
  std::cout << "Operator: "
            << (scheduler_options.busy_wait ? "BusyWaitCountOp (1 ms busy wait, 1,000 ops/operator)"
                                            : "CountOp (100,000 ops/operator)")
            << '\n';
  std::cout << fmt::format("Scheduler options: queue_stealing={}, postcheck_fastpath={}",
                           scheduler_options.enable_queue_stealing ? "on" : "off",
                           scheduler_options.enable_postcheck_fastpath ? "on" : "off")
            << '\n';
  std::cout << fmt::format("\n| {:>5} | {:>7} | {:>9} | {:>10} | {:>12} |",
                           "Trial",
                           "Threads",
                           "Operators",
                           "Total Ops",
                           "Ops/s")
            << '\n';
  std::cout << "|-------|---------|-----------|------------|--------------|" << '\n';

  for (size_t i = 0; i < trial_options.size(); i++) {
    BenchmarkSchedulerThroughputApp::Options& options = trial_options[i];
    BenchmarkSchedulerThroughputApp::Results& results = trial_results[i];

    std::cout << fmt::format("| {:>5} | {:>7} | {:>9} | {:>10} | {:>12.1f} |",
                             i,
                             options.num_threads,
                             options.num_operators,
                             results.total_operations,
                             results.throughput_hz)
              << '\n';
  }

  return 0;
}
