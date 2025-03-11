from setuptools import setup, find_packages

setup(
    name='temporal_norm',
    version='0.1',
    description='experiments for multi source sleep staging',

    # The project's main homepage.
    # url='https://github.com/tgnassou/da-toolbox',

    # Author details
    author='Théo Gnassounou',
    author_email='theo.gnassounou@inria.fr',

    # Choose your license
    license='BSD 3-Clause',
    # What does your project relate to?
    keywords='sleep staging, multi-source, domain adaptation',

    packages=find_packages(),
)
