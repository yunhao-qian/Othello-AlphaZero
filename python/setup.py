from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="othello-alpha-zero",
    version="0.1.0",
    install_requires=[],
    packages=find_packages(include=["othello_alpha_zero", "othello_alpha_zero.*"]),
    ext_modules=cythonize("othello_alpha_zero/**/*.pyx"),
)
