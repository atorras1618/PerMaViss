""" Setup script. """

from setuptools import setup, find_packages

exec(open("src/permaviss/version.py", "r").read())

setup(
    name="permaviss", 
    version=__version__
    url="https://github.com/atorras1618/PerMaViss",
    author="Alvaro Torras Casas",
    licence="MIT",
    packages=find_packages("src")
)
