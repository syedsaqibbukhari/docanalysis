# README file for Page Cropping component

Filename : ocrd-anyBaseOCR-pagecropping.py

Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
Note: 
1) this work has been done in DFKI, Kaiserslautern, Germany.
2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# Method Behaviour 
 This function takes a document image as input and crops/selects the page content area only (that's mean remove textual noise as well as any other noise around page content area)


# LICENSE 
 Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
 Apache License 2.0

 A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.


# Usage:
python ocrd-anyBaseOCR-cropping.py -m (path to met input file) -I (Input group name) -O (Output group name) -w (Working directory)
	[-p (path to parameter file) -o (METs output filename)]

# Example: 
python ocrd-anyBaseOCR-cropping.py -m work_dir/mets.xml -I OCR-D-IMG-DESKEW -O OCR-D-IMG-CROP -w work_dir
