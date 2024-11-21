from setuptools import setup, find_packages

setup(
    name="spaceship_preprocessing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn'
    ]
)