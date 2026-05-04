import os
from typing import List, Tuple
from urllib.request import Request

import requests
from requests import HTTPError, RequestException


def fake_gcs_server_is_alive():
    try:
        resp = requests.get("http://127.0.0.1:4443/storage/v1/b")
        return (resp.status_code == 200)
    except ConnectionError:
        return False
    except TimeoutError:
        return False
    except HTTPError:
        return False
    except RequestException:
        return False
    

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
    return (os.path.join(base_dir, "ratings_train_liked.array_record"),
        os.path.join(base_dir, "ratings_val_liked.array_record"), os.path.join(base_dir, "ratings_test_liked.array_record"))
    
def get_train_val_test_disliked_uris(use_small:bool=True) -> Tuple[str, str, str]:
    base_dir = os.path.join(get_project_dir(), "src/test/resources/data/")
    if use_small:
        base_dir = os.path.join(base_dir, "small")
    return (os.path.join(base_dir, "ratings_train_disliked.array_record"),
        os.path.join(base_dir, "ratings_val_disliked.array_record"), os.path.join(base_dir, "ratings_test_disliked.array_record"))
