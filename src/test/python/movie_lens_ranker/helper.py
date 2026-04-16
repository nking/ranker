import os
from typing import List, Tuple


def get_kaggle() -> bool:
  cwd = os.getcwd()
  if "kaggle" in cwd:
    kaggle = True
  else:
    kaggle = False
  return kaggle

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

def get_train_val_test_liked_uris(use_small:bool=True) -> Tuple[str, str, str]:
    base_dir = os.path.join(get_project_dir(), "src/test/resources/data/")
    if use_small:
        base_dir = os.path.join(base_dir, "small")
    return (os.path.join(base_dir, "ratings_train_liked"),
        os.path.join(base_dir, "ratings_val_liked"), os.path.join(base_dir, "ratings_test_liked"))
    
def get_train_val_test_disliked_uris(use_small:bool=True) -> Tuple[str, str, str]:
    base_dir = os.path.join(get_project_dir(), "src/test/resources/data/")
    if use_small:
        base_dir = os.path.join(base_dir, "small")
    return (os.path.join(base_dir, "ratings_train_disliked"),
        os.path.join(base_dir, "ratings_val_disliked"), os.path.join(base_dir, "ratings_test_disliked"))
