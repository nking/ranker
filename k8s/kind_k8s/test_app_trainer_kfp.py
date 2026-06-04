import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unittest
from unittest import TestCase
from run_app_trainer_kfp import compile_pipeline_yaml

def get_project_dir() -> str:
    cwd = os.getcwd()
    head = cwd
    proj_dir = ""
    while head and head != os.sep:
        head, tail = os.path.split(head)
        if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
            if tail == "ranker":
                proj_dir = os.path.join(head, tail)
                break
    return proj_dir

def get_bin_dir() -> str:
    return os.path.join(get_project_dir(), "bin")

class Test(TestCase):
    def test_compile_pipeline_yaml(self):
        train_job_yaml_path = os.path.join(get_project_dir(), "k8s/kind_k8s/train_job.yaml")
        output_path = os.path.join(get_bin_dir(), "output_pipeline.yaml")
        print(f'writing to {output_path}', flush=True)
        compile_pipeline_yaml(output_path, train_job_yaml_path, num_trials=4)

if __name__ == '__main__':
    unittest.main()
