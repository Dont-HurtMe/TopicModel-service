import os
import re
from setuptools import setup, find_packages

package = "dsm_wizservices"

def get_version():
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)

def read_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='dsm_wizservices',
    version=get_version(),
    description='A service for topic modeling using visualize sentence embedding.',
    author_email='pwlnwzarediooo@gmail.com',
    url='https://github.com/Dont-HurtMe/TopicModel-service',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),
    dependency_links=[
        'git+https://github.com/Dont-HurtMe/TopicModel-service.git#egg=TopicModelService-0.1.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.10',
)
