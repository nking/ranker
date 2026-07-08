load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "graph_builder_op.so",
    srcs = ["graph_builder_op.cc"],
    linkshared = True, # This tells Bazel to build a .so file, not an executable
    # deps = ["@tensorflow_headers//:tensorflow"] # You will configure this linkage soon
)