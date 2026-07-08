creates a C++ gRPC service for using the ScANN library in a development
environment while no access to Vertex AI Vector Search
or Alloy DB, nor Amazon OpenSearch, nor Azure AI Search

software to download or install.
- MS VSCode
  - and C/C++ dev extension by MS
  - and Bazel extension by Google/BazelTeam
  - and Dev Containers extension by MS
- the TF container that has bazel build tools installed.
  docker pull us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest

open VSCode
open folder ranker/scann_c++_grpc_service_src
mkdir .devcontainer
vi ranker/scann_c++_grpc_service_src/.devcontainer/devcontainer.json
and put into it the following:
{
  "name": "ScaNN C++ gRPC Builder",
  "image": "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-vscode.cpptools",
        "bazelbuild.vscode-bazel"
      ]
    }
  },
  "remoteUser": "root" 
}
# that will automatically install the C++ and Bazel extension into the container

in VSCode 
   Open the Command Palette by pressing Ctrl + Shift + P (or Cmd + Shift + P on Mac).
   Type "Dev Containers: Reopen in Container" and hit Enter.
   when finished the explorer window will show 
      Dev Container: TF Custom Op Builder
   press <ctr> and ` at same time to get a prompt to a terminl running
      inside the ai-build (bazel) container
   touch WORKSPACE
   skipping the rest of the directions
   create BUILD and  put in it:
      load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "graph_builder_op.so",
    srcs = ["graph_builder_op.cc"],
    linkshared = True, # This tells Bazel to build a .so file, not an executable
    # deps = ["@tensorflow_headers//:tensorflow"] # You will configure this linkage soon
)

create graph_builder_op.cc

bazel build //:graph_builder_op.so





