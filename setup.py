__author__ = 'leonardtruong'

from setuptools import setup, find_packages

setup(
    name='hindemith',
    version='0.0.2a3',
    author='Leonard Truong',
    author_email='leonardtruong@berkeley.edu',
    description='Package containing a suite of high performance pattern specializers',
    url='https://github.com/ucb-sejits/hindemith',

    packages=find_packages(),

    install_requires=[
        'ctree',
        # 'pycl',
        'numpy'
    ]
)
