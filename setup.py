from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['Horizon', 'constraints', 'solvers', 'utils'],
    package_dir={'Horizon': 'Horizon',
		 'constraints': 'Horizon/constraints',
		 'solvers': 'Horizon/solvers',
		 'utils': 'Horizon/utils'}
)

setup(**d)
