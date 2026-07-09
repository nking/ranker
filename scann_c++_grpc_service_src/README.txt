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
had to create .devcontainer and its contents
# that will automatically install the C++ and Bazel extension into the container

in VSCode 
   Open the Command Palette by pressing Ctrl + Shift + P (or Cmd + Shift + P on Mac).
   Type "Dev Containers: Reopen in Container" and hit Enter.
   when finished the explorer window will show 
      Dev Container: SCANN_C++_GRPC_SERVICE_SRC
   press <ctr> and ` at same time to get a prompt to a terminl running
      inside the ai-build (bazel) container

   create BUILD  file
   create MODULE.bazel  file

   see files...

In the dev container terminal window, install buildifier:
# Replace <version> with the latest (e.g., 8.5.1)
wget https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-linux-amd64
chmod +x buildifier-linux-amd64
sudo mv buildifier-linux-amd64 /usr/local/bin/buildifier

download bazelisk (needed to use bazel server version 8.7.0
in the dev container terminal, use:
wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
create file .bazelversion and put 8.7.0 in it
bazel version
echo "common --enable_workspace" > .bazelrc

building ScANN requires a few files from tensorflow like
libtensorflow_framework.so* and protocol buffer files from
a build (like tensorflow/core/framework/full_type.pb.h)
so we can use a pip installed tensorflow for that.
    pip install tensorflow==2.20.0

needed to add to MODULE.bazel, an entry for envoy_api
needed to createWORKSPACE  entry for scann

see files ...

to build project:
in dev container terminal type:
rm -f MODULE.bazel.lock
bazel clean --expunge
bazel build //:scann_server

helpful in debugging:
   bazel query "kind(cc_library, @scann//scann/...)"
   bazel info output_base
   du -sh $(bazel info output_base)/external
   bazel mod graph --depth=10
   bazel mod tidy
   bazel query "@eigen//..."

to check BUILD files
    buildifier BUILD

