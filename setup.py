__author__ = 'leonardtruong'

from setuptools import setup, find_packages

setup(
    name='hindemith',
    version='0.0.1a3',
    author = 'Leonard Truong',
    author_email = 'leonardtruong@berkeley.edu',

    packages = find_packages(),

    install_requires=[
        'ctree',
        # 'pycl',
        'numpy'
    ]
)
