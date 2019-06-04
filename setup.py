# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = 'ocrd-anybaseocr',
    version = 'v0.0.1',
    author = "Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib",
    author_email = "Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de",

    url = "https://github.com/syedsaqibbukhari/docanalysis",

    license='Apache License 2.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').read().split('\n'),
    packages=find_packages(exclude=["work_dir","models","pix2pixHD_modified"]),
    package_data={
        '': ['*.json']
    },
    entry_points={
        'console_scripts': [
            'ocrd-anybaseocr-binarize = ocrd_anybaseocr.cli.ocrd_anybaseocr_binarize:main',
            'ocrd-anybaseocr-crop     = ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping:main',
            'ocrd-anybaseocr-deskew   = ocrd_anybaseocr.cli.ocrd_anybaseocr_deskew:main',
            'ocrd-anybaseocr-dewarp   = ocrd_anybaseocr.cli.dewarp:main'
        ]
    },
)