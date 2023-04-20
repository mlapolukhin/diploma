#!/usr/bin/env python

from setuptools import setup, find_packages


name = "diploma"
version = "0.0.1"
description = "Diploma Research"
author = "Andrii Polukhin"

setup(
    name=name,
    version=version,
    description=description,
    author=author,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
