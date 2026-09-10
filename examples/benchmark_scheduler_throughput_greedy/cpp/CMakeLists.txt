# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(benchmark_scheduler_throughput_greedy
  benchmark_scheduler_throughput_greedy.cpp
)

target_link_libraries(benchmark_scheduler_throughput_greedy
  PRIVATE
  holoscan::core
)

file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(benchmark_scheduler_throughput_greedy PROPERTIES
                      INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
  install(FILES benchmark_scheduler_throughput_greedy.cpp
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )

  install(FILES CMakeLists.min.txt
    RENAME "CMakeLists.txt"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )
endif()

install(TARGETS benchmark_scheduler_throughput_greedy
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_GREEDY_TEST
           COMMAND benchmark_scheduler_throughput_greedy
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_GREEDY_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "GreedyScheduler Throughput Benchmark Results"
                       TIMEOUT 120)

  add_test(NAME EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_GREEDY_BUSY_WAIT_TEST
           COMMAND benchmark_scheduler_throughput_greedy --busy_wait
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_GREEDY_BUSY_WAIT_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Operator: BusyWaitCountOp"
                       TIMEOUT 120)

  add_test(NAME EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_GREEDY_HELP_TEST
           COMMAND benchmark_scheduler_throughput_greedy --help
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_GREEDY_HELP_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Usage:"
                       TIMEOUT 10)
endif()
