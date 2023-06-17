from setuptools import setup, find_packages

setup(
    name='Double Dobust did',
    version='0.2.0',
    description='Descripción de la librería',
    url='https://github.com/d2cml-ai/drdid',
    author='Jhon Flores',
    author_email='fr.jhonk@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas', 
        'numpy<=1.24.3',
        'statsmodels'
    ]
)