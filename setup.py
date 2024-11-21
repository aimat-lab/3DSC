#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [req for req in requirements if not req.startswith("#")] # remove outcommented requirements

setup(
    name='superconductors_3D',
    version='0.1.0',
    description='Code of the 3DSC paper.',
    author='Timo Sommer, Pascal Friederich',
    python_requires=">=3.8.6",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'make_3DSC=superconductors_3D.cli:main',
        ],
    },
    )