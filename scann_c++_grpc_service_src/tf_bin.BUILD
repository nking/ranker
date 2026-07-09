# --- tf_bin.BUILD ---
package(default_visibility = ["//visibility:public"])

# Expose the .so framework file found inside the pip installation
filegroup(
    name = "lib_so",
    srcs = glob(["**/libtensorflow_framework.so*"]),
)

cc_library(
    name = "eigen_bridge",
    deps = ["@eigen//:eigen"], 
    includes = ["."], 
    visibility = ["//visibility:public"],
)

# Expose the pre-compiled headers (including the missing .pb.h files)
cc_library(
    name = "tf_bin_headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp",
       "include/**/*.inc", 
       #"include/unsupported/**/*",
       ]),
    includes = ["include"],
    deps = [":eigen_bridge"], # Connect the bridge
    visibility = ["//visibility:public"],
)
