from setuptools import setup, find_packages

from drdid._version_ import version

setup(
    name='Double Dobust did',
    version=version,
    description='Descripción de la librería',
    url='https://github.com/d2cml-ai/drdid',
    author='Jhon Flores',
    license="MIT",
    author_email='fr.jhonk@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas', 
        'numpy<=1.24.3',
        'statsmodels'
    ]
)