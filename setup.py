from setuptools import setup, find_packages

setup(
    name='hindemith',
    version='0.1.0',
    author='Leonard Truong',
    author_email='leonardtruong@berkeley.edu',
    description='',
    url='https://github.com/ucb-sejits/hindemith',

    packages=find_packages(exclude=['examples', 'tests']),

    install_requires=[
        'ctree',
        'pycl',
        'numpy'
    ]
)
