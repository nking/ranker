# ranker

project to create a ranker model for the mid-tier
in a 3 part recommendation system.  Part 1 is
the Two-tower bi-encoder + Scann for retrieval.
part 2 is the ranker (list-wise cross-encoder) in this project.
part 3 is the re-ranker (fine-tuned pretrained list-wise cross-encoder).

this project will create a ranker from
the Jax AI Stack + Rax.
It will use a kubeflow pipeline for an MlOps pipeline

part 1 can be found at:
https://github.com/nking/recommender_systems.git
and
https://github.com/nking/retrieval.git

part 3 can be found at:
https://github.com/nking/reranker.git

instructions:
  set up a virtual environment using conda or virtualenv
  with python version 3.12 
     e.g. conda create --name ranker_py312 python=3.12
  
  activate the virtual environment
     e.g. conda activate ranker_py312

to run the unit tests:
(1) generate the data:
    The data are only present as DVC commits because some are
    large files.  The files can be recreated following notes
    in src/test/resources/README.txt
    TODO: reconsider committing them in this project
(2) install the dependencies
    the easiest way is to install this project:
        pip install --editable .
    else you can find the required libraries in pyproject.toml
    or setup.py or requirements.txt
(3) install docker or equivalent and start it
    see https://download.docker.com and instructions
(4) prepare input data from step (1) for reading by local blob storage:
    cd scripts
    sh < prep_for_tests.sh
(5) ... docker-compose.yaml...
   docker compose -f docker-compose.yaml build app
   docker compose -f docker-compose.yaml run --rm app

-- currently just testing integration of all services
- for xmanager,
   will need to have docker-compose-data.yaml run before the
   piepline, already existing,
  and it won't include the train_fn app.

Local testing:

  pycharm:

    using right click menu, mark the source tree directory:
      src/main/python

    using right click menu, mark the test tree directory:
      src/test/python/movie_lens_ranker

    then pycharm tests will correctly resolve paths.

  bash or other shell environment:

    python and pytest can be used from the project's base
    directory

to build the docker image and run the container locally:
.. in progress
  docker compose up

and when done:
#stop process, but keep containers using:
  docker compose stop
#stop process, remove containers, keep volumne safe
  docker compose down

--------------------------------------------------
some details about the model

- base layer an instance of jraphx.nn.GATv2Conv
  which is the jraphx implementation of the
  Brody, Alon, Yahav "How Attentive are Graph Attention Networks"
  model called "Universal Graph Attention".
  - has dynamic attention that is dependent upon the query.
  - complexity is O(V*F + E*F) where
    V is number of nodes, E is number of edges and F is number of
    features.
  - e(h_i, h_j) = a^T * leaky_relu(W * (h_i concat h_j))
    where a is the attention vector.
  TODO: finish details here...

  TODO: considering a CliffordNet version, but it needs
        a bi-encoder that uses geometric algebra
        and the retrieval needs adaptations.
