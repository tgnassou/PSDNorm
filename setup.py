from setuptools import setup, find_packages

setup(
    name='temporal_norm',
    version='0.1',
    description='Experiments for test time temporal normalization',

    # Author details
    author='Théo Gnassounou',
    author_email='theo.gnassounou@inria.fr',

    # Choose your license
    license='BSD 3-Clause',
    # What does your project relate to?
    keywords='sleep staging, test time, normalization layer',

    packages=find_packages(),
)
