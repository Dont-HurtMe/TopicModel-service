# setup.py
import os
from setuptools import setup, find_packages

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='wkge-service',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
        # 'somepackage>=1.0',
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here.
            # 'mycommand = myproject.module1:main',
        ],
    },
    description='A simplified implementation of WizMap for document clustering and summarization using LLMs.',
    # long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',],
    python_requires='>=3.10',
)
