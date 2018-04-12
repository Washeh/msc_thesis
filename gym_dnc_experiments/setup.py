from setuptools import setup

setup(name='gym_dnc_experiments',
      version='0.0.1',
      packages=["gym_dnc_experiments", "gym_dnc_experiments.envs"],
      install_requires=['gym>=0.9.4', 'numpy>=1.14.0']
)
