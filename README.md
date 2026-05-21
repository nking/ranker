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
    see scripts/init_ranker_venv.sh 
(3) install docker or equivalent and start it
    see https://download.docker.com and instructions
(4) prepare input data from step (1) for reading by local blob storage:
    cd scripts
    sh < prep_for_tests.sh
(5) there are a couple of ways to start the services depending on what
    the goal is:
    (a) to run test_Ranker.py test_run_train_with_optuna
        you can start the db services alone:
           docker compose -f docker-compose-dbs.yaml up -d --build
        then run the tests
        then when done:
           docker compose down
    (b) to start the dbs and the container app:
            docker compose -f docker-compose.yaml build app
        or 
            docker compose -f docker-compose.yaml build --no-cache app
        then to run a trial train with a small sample over 2 epochs: 
            sh < check_can_run.sh

    NOTE that once the app image is built, no need to rebuild the image.
    changing parameters in the docker-compose*.yaml can be run with:
       docker compose up -d
    if you update code in the app service:
       docker compose up -d --build
    if have trouble cleaning up networks use
       docker compose down
       docker compose up -d

- there is a unittest called test_Ranker.py which is an integration
  test of app_rnner for a single process single host environment.
- there is a multi-host, multi-process test in
  xmngr/launcher_pipeline.py
  it requires a separate venv to install xmanager into.
  see scripts/init_xmanager_venv.sh
  NOTE: on a single CPU, it is better to run using jax process count = 1
  due to expenses of context switching and communication overhead for this app.  
  The script tests that the code
  would still function correctly if 2 of the CPU's cores are used.
  There are use cases when jax process count > 1 are a good idea.

- there will be a kubeflow pipeline using Trainer API v2

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
  - The model takes in a user's query graph which is the user query
    enriched by the user's watch history and candidate movies.
    During training, the candidate movies are negatives for the
    constrastive listwise learning.
    During inference, the candidate movies are from the retrieval
    stage of the recommendation system.

    The GATV2 takes the embeddings created by the bi-encoder and
    takes the ratings as edge attributes, and learns a weight matrix
    over a specified number of hops in the query graph network
    nodes, and learns an edge weight matrix, and learns the attention
    that node_i pays to node_j.
    The learning is about interactions between elements in the embeddings
    rather than interactions between ids, and so the results are inductive
    and can be applied to inputs that have not been seen before.

    The GATv2 is followed by a linear layer with output dimension 1 in order
    to produce scores for each candidate.

    The results are useful relative to the input graph, that is, for ranking
    the candidates of the user query.

  TODO: considering a CliffordNet version, but it needs
        upstream of it, a bi-encoder that uses geometric algebra
        and it requires adaptations to the retrieval.
        - this Geometric Algebra (GA) cross encoder would have layers:
          (1) a geometric product layer to characterize the interaction
             where the GP is dot product + wedge product between a
             query and candidate bi-vectors
          (2) a linear layer to characterize scaling and shear and also
              an adapter between the output of layer 1 and input of layer 3
          (3) a rotor layer to characterize rotation of embedding axes
              and replace the GATv2 attention.
        presumably I can use a smaller number of axes in the
          input bivector embeddings (n=6 bivector as comparable to the 
          length=16 euclidean vector, considering the combinatorial C(n, 2))
          and get a more expressive result for similar final runtime complexity
          and space complexity.
          ==> that the runtime and space complexities are of same order as
              the prototype GraphRanker means that the model would need to
              train in a cluster, preferably with accelerators.
