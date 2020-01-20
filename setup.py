""" Setup script. """

from setuptools import setup, find_packages

from src.permaviss.version import __version__

with open("README.rst", "r") as readme_file:
    README = readme_file.read()

setup(
    name="permaviss",
    version=__version__,
    description="Persistence Mayer Vietoris spectral sequence",
    long_description=README,
    long_description_content_type="text/x-rst",
    author_email="atorras1618@gmail.com",
    author="Alvaro Torras Casas",
    url="https://github.com/atorras1618/PerMaViss",
    download_url="https://github.com/atorras1618/PerMaViss/tarball/v0.0.1",
    licence="MIT",
    keywords=["spectral sequence", "persistent homology", "Mayer Vietoris"],
    packages=find_packages("src"),
    package_dir={"": "src"},
)
