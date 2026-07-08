load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

# Define the language-agnostic proto library
proto_library(
    name = "scann_proto",
    srcs = ["service.proto"],
)

# Define the C++ specific gRPC library
# This rule takes the proto and generates the C++ boilerplate automatically
cc_grpc_library(
    name = "scann_cc_grpc",
    srcs = [":scann_proto"],
    grpc_only = False, # Set to False to get both Protobuf messages AND gRPC stubs
    deps = [":scann_proto"],
)

# this library
cc_library(
    name = "searcher_lib",
    srcs = ["searcher.cc"],
    hdrs = ["searcher.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@scann//scann:scann_ops",
    ],
)

# Use it in your binary
cc_binary(
    name = "scann_server",
    srcs = ["main.cc"],
    deps = [
        ":searcher_lib",
        "@com_github_grpc_grpc//:grpc++",
    ],
)