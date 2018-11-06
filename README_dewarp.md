====================================
README file for Dewarping component
====================================

Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
Note: 
1) this work has been done in DFKI, Kaiserslautern, Germany.
2) At the moment there are no exposed parameters that can be changed.
3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

*********** Method Behaviour ********************
# This function takes a document image as input and make the text line straight if its curved.
*********** Method Behaviour ********************

*********** LICENSE ********************
# Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Apache License 2.0

# A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

*********** LICENSE ********************

Usage:
python ocrd-anyBaseOCR-dewarp.py -m (path to met input file) -I (Input group name) -O (Output group name) -w (Working directory)
	[-p (path to parameter file) -o (METs output filename)]

Example: 
python ocrd-anyBaseOCR-dewarp.py -m work_dir/mets.xml -I OCR-D-IMG-CROP -O OCR-D-IMG-DEWARP -w work_dir
