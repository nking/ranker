# ranker
project to create a ranker model for the mid-tier
in a 3 part recommendation system.  part 1 is
the Two-tower bin-encoder + Scann for retrieval.
part 2 is the ranker in this project.
part 3 is the re-ranker.

this project will create a ranker from
Jraph + Rax
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
  or setup.py

  the unit tests show how to run the code.

Local testing:

  pycharm:

    using right click menu, mark the source tree directory:
      src/main/python

    using right click menu, mark the test tree directory:
      src/test/python/movie_lens_retrieval

    then pycharm tests will correctly resolve paths.

  bash or other shell environment:

    python and pytest can be used from the project's base
    directory

--------
versions
rax 0.4.0 was release jan 7, 2025 so might need jax v0.4.38
