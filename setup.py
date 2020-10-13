from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['Horizon', 'Horizon.constraints', 'Horizon.solvers', 'Horizon.utils'],
    package_dir={'Horizon': 'Horizon',
		 'Horizon.constraints': 'Horizon/constraints',
		 'Horizon.solvers': 'Horizon/solvers',
		 'Horizon.utils': 'Horizon/utils'}
)

setup(**d)
