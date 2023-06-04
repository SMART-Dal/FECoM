# this is executed when you run "pip install ."

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='greenAI-tool',
    version='0.0.0',
    description='GreenAI-extension-tool',
    long_description=readme,
    author='Tim Widmayer',
    author_email='tim.widmayer.20@ucl.ac.uk',
    url='https://github.com/mkechagia/GreenAI-extension/tree/main/tool',
    # license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data', 'deprecated', 'replication'))
)