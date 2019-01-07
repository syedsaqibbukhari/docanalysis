from __future__ import print_function

import sys
import glob
import os.path
from distutils.core import setup


ocrolib = [c for c in glob.glob("ocrolib/*")]
scripts = [c for c in glob.glob("ocrd-anyBaseOCR-*")]

setup(
    name = 'anyBaseOCR',
    version = 'v0.0.1',
    author = "Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib",
    author_email = "Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de",
    url = "https://github.com/syedsaqibbukhari/docanalysis",
    license = "Apache License 2.0",
    description = "Binarize, Deskew, Cropping OCR-D historical document images",
    packages = ["ocrolib"],
    scripts = scripts,
)