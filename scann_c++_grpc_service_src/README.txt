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
      Dev Container: SCANN_C++_GRPC_SERVICE_SRC
   press <ctr> and ` at same time to get a prompt to a terminl running
      inside the ai-build (bazel) container

   create BUILD  file
   create MODULE.bazel  file

   see files...


In the terminal window, install buildifier:
# Replace <version> with the latest (e.g., 8.5.1)
wget https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-linux-amd64
chmod +x buildifier-linux-amd64
sudo mv buildifier-linux-amd64 /usr/local/bin/buildifier

then in the terminal, use:

to resolve MODULE.bazel dependency tree and download files:
    bazel mod deps

shows that ScANN project hasn't written a valid Bzlmod yet,
so need to work around that using a hybrid approach using
the legacy WORKSPACE file.

to build project:
    buildifier BUILD

