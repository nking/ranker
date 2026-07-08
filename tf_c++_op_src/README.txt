NOTE: this has been abandoned to use the rust graph_builder instead.

The notes are left here as they useful in any case.

this is a TF C++ operation for the graph builder needed to receive
a query input of user_id, user_embedding and candidate_embeddings
(as tf pointers to memore of tf tensors) and
return the 9 arrays needed for input to GraphRanker.

The bigger picture was to make a single TF execution graph
encompassing as one SavedModel
the use of the Keras2 TwoTower SavedModel query signature,
a preloaded and cached Scann Indexer, the TF c++ op
graph builder from this directory, and the
JAX AI stack GraphRanker SavedModel.

software to download or install.
- MS VSCode
  - and C/C++ dev extension by MS
  - and Bazel extension by Google/BazelTeam
  - and Dev Containers extension by MS
- the TF container that has bazel build tools installed.
  choose one of:
    docker pull us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest
  or
    docker pull tensorflow/build:latest-python3.12
  or
    docker pull tensorflow/build:latest-cuda12.3-cudnn8.9-amd64

open VSCode
open folder ranker/tf_c++_op_src
mkdir .devcontainer
vi ranker/tf_c++_op_src/.devcontainer/devcontainer.json
and put into it the following:
{
  "name": "TF Custom Op Builder",
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

see code...

SHUTTING DOWN the VSCode dev container:
use command pallette (lower left corner, settings icon)
--> type into the box: 
     Close Remote Connection
    select it.
    then the window clears for a new project
if it's running in docker stop it:
# List containers to find your old one
docker ps -a
# Stop and remove the container
docker stop <container_id>
docker rm <container_id> 

