import os
from setuptools import setup, find_packages

package = "dsm_wizservices"

def read_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as readme:
    README = readme.read()

setup(
    name='dsm_wizservices',
    version='0.0.1',
    description='A service for topic modeling using visualize sentence embedding.',
    long_description=README,
    long_description_content_type='text/markdown',
    author_email='pwlnwzarediooo@gmail.com',
    url='https://github.com/Dont-HurtMe/TopicModel-service',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.9',
)
