# this is executed when you run "pip install ."

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='FECoM-tool',
    version='1.0.0',
    description='The FECoM tool (Fine-grained Energy Consumption Measurement) can measure energy consumption of Python code at method-level precision, in particular for machine learning with TensorFlow',
    long_description=readme,
    author='FECoM-Authors',
    license=license,
    packages=find_packages(exclude=('data', 'replication'))
)