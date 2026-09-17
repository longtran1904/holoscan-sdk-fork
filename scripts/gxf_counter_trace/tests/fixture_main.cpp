/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include "gxf/core/gxf.h"

int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  const char* mode = argv[1];
  const int repeats = std::strcmp(mode, "sequential") == 0 ? 2 : 1;
  for (int i = 0; i < repeats; ++i) {
    gxf_context_t context{};
    GxfContextCreate(&context);
    auto result = GxfGraphRunAsync(context);
    if (std::strcmp(mode, "run_failure") == 0) {
      if (result != GXF_FAILURE)
        return 3;
    } else {
      if (result != GXF_SUCCESS)
        return 4;
      if (std::strcmp(mode, "concurrent") == 0) {
        gxf_context_t second{};
        GxfContextCreate(&second);
        GxfGraphRunAsync(second);
        GxfGraphWait(second);
        GxfContextDestroy(second);
      }
      if (std::strcmp(mode, "abort") == 0) {
        std::puts("fixture ready");
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(30));
      } else if (std::strcmp(mode, "destroy") == 0) {
        GxfContextDestroy(context);
        continue;
      } else if (std::strcmp(mode, "deactivate") == 0) {
        GxfGraphDeactivate(context);
      } else {
        result = GxfGraphWait(context);
        if (result != (std::strcmp(mode, "wait_failure") == 0 ? GXF_FAILURE : GXF_SUCCESS))
          return 5;
        if (std::strcmp(mode, "wait_failure") != 0)
          GxfGraphDeactivate(context);
      }
    }
    GxfContextDestroy(context);
  }
  return 0;
}
