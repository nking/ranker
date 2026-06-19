from setuptools import setup, find_packages

setup(
  name='movie_lens_ranker',
  version='0.1.0',
  packages=find_packages(where="src/main/python",
    include=['movie_lens_ranker']),
  package_dir={'': 'src/main/python'},
  #  RUN pip install --no-deps -r requirements-cpu.txt or -gpu.txt
  install_requires = [
  ],
  #extras_require={"test": ["pytest"]},
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
