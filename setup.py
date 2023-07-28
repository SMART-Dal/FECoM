# this is executed when you run "pip install ."

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='greenAI-tool',
    version='1.0.0',
    description='GreenAI tool to measure energy consumption of Python code, in particular for machine learning with TensorFlow',
    long_description=readme,
    author='Tim Widmayer',
    author_email='tim.widmayer.20@ucl.ac.uk',
    url='https://github.com/mkechagia/GreenAI-extension/tree/main/tool',
    license=license,
    packages=find_packages(exclude=('data', 'deprecated', 'replication'))
)