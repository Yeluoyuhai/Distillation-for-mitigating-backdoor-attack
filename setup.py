import os
import sys
from setuptools import setup, find_packages

requirements = [
  "numpy",
  "matplotlib",
  "seaborn",
  "scikit-learn",
  "pysmb",
  "opencv-python",
  "jupyter",
  "setuptools",
  "torch",
  "torchvision",
  "pytorch-ignite",
  "pytorch-pfn-extras"
]

setup(
  name='backdoor attack',
  version='0.0.1',
  description='backdoor attack and defences with distillation',
  long_description="",
  author='Kota Yoshida',
  author_email='ri0044ep@ed.ritsumei.ac.jp',
  install_requires=requirements,
  url='',
  license=license,
  packages=find_packages(exclude=('tests', 'docs')),
  extras_require={
    'dev':[
    ]
  }
)