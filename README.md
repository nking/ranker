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

part 3 can be found at:
https://github.com/nking/reranker.git

instructions:
  set up a virtual environment using conda or virtualenv
  with a python version that is >= 3.10.0
     e.g. conda create --name ranker_py312 python=3.12
  
  activate the virtual environment
     e.g. conda activate ranker_py312

  to install the dependencies, the easiest way is to
  install this project:
    pip install --editable .
  else you can find the required libraries in pyproject.toml
  or setup.py or requirements.txt

  the unit tests show how to run the code.
  The data are only present as DVC commits because some are
  large files.  The files can be recreated following notes
  in src/test/resources/README.txt

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

--------------------------------------------------
some details about the model

- base layer an instance of jraphx.nn.GATv2Conv
  which is the raphx implementation of the
  Brody, Alon, Yahav "How Attentive are Graph Attention Networks"
  model called "Universal Graph Attention".
  - has dynamic attention that is dependent upon the query.
  - complexity is O(V*F + E*F) where
    V is number of nodes, E is number of edges and F is number of
    features.
  - e(h_i, h_j) = a^T * leaky_relu(W * (h_i concat h_j))
    where a is the attention vector.
