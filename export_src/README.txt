This is a project to export the JaXAI stack model.
It needs tensorflow to define the signatures, so needs
a separate venv.

can clone the existing development venv:
   conda create --name ranker_tf_py312 python=3.12
   
   conda activate ranker_tf_py312

   #cd to base directory
   cd ../
   pip install --no-deps -r requirements-cpu.txt
   pip install --no-deps -e . 
   
   #cd back to this directory
   cd export_src
   pip install --no-deps -r requirements.txt

   pip install --no-deps -e . 

after running test_export.py, can inspect the savedmodel:
   saved_model_cli show --dir ../bin/savedmodels/1 --all
