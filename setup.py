from setuptools import setup, find_packages

setup(
  name='movie_lens_ranker',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_ranker']),
  package_dir={'': 'src/main/python'},
  install_requires = [
    'rax==0.4.0', 'jax-ai-stack==2025.10.28', 'jraphx==0.0.4',
    'jraph @ git+https://github.com/deepmind/jraph.git@51f5990104f7374492f8f3ea1cbc47feb411c69c',
    'dvc==3.67.0'
  ],
  extras_require={"test": ["pytest"]},
  classifiers=[ 'Natural Language :: English',
               'Programming Language :: Python :: 3.12',
               'Development Status :: 1 - Development/Unstable'
  ],
  url='https://www.kaggle.com/code/nicholeasuniquename/ranker/',
  license='MIT',
  author='Nichole King',
  author_email='',
  description='Ranking for Kaggle recommender systems project'
)
