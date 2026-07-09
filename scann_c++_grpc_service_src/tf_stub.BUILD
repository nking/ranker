# --- tf_stub.BUILD ---
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = glob(["**/*.h"]),
    includes = ["."],
    # Bridge to the local pip installation's pre-compiled headers
    deps = ["@tf_local_pip//:tf_bin_headers"],
)

cc_library(
    name = "libtensorflow_framework",
    # Bridge to the local pip installation's compiled library
    srcs = ["@tf_local_pip//:lib_so"],
    deps = [":tf_header_lib"],
)

cc_library(
    name = "tf_run_main",
    srcs = [],
)
