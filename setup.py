from setuptools import setup, find_packages

from drdid._version_ import version

setup(
    name='drdid',
    version=version,
    description='Double Robust Difference in Difference y python',
    url='https://github.com/d2cml-ai/drdid',
    author='Jhon Flores',
    license="MIT",
    author_email='fr.jhonk@gmail.com',
    packages=['drdid'],
    install_requires=[
        'pandas', 
        'numpy<=1.24.3',
        'statsmodels'
    ],
    long_description='''
    Implementation of drdid for Python, R-like syntax with return destructuring using optimized Python code.

    See the original [R package](https://github.com/pedrohcgs/DRDID/tree/master)
    '''
)