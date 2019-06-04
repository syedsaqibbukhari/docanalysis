# ========================================================================
# README file for Dewarping component
# ====================================

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note: 
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). 
# Two sample image file (samples/becker_quaestio_1586_00013.tif; samples/estor_rechtsgelehrsamkeit02_1758_0819.tif) and mets.xml (work_dir/mets.xml) are provided. 
# The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and make the text line straight if its curved.
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Apache License 2.0

# A permissive license whose main conditions require preservation of copyright 
# and license notices. Contributors provide an express grant of patent rights. 
# Licensed works, modifications, and larger works may be distributed under 
# different terms and without source code.

# *********** LICENSE ********************
# ========================================================================

#! /usr/bin/env python

import sys, os, argparse
import numpy
from numpy import *
from scipy.ndimage import gaussian_filter1d
import copy
import pylab
from pylab import find , plot, matshow, show, arctan
import PIL
import PIL.Image
import PIL.ImageFilter
import PIL.Image , PIL.ImageDraw
import re
import string
import math
from math import ceil, e, sqrt
from scipy import median
from re import split
import time
from time import clock,time
import cv2
import json
from xml.dom import minidom
import  torch
from ParserAnybaseocr import *

def dewarping(tmp, dest):
	os.system("python pix2pixHD/test.py --dataroot %s --checkpoints_dir ./ --name models --results_dir %s --label_nc 0 --no_instance --no_flip --resize_or_crop none --n_blocks_global 10 --n_local_enhancers 2 --gpu_ids %s --loadSize %d --fineSize %d --resize_or_crop %s" % (os.path.dirname(tmp), dest, args.gpu_id, args.resizeHeight, args.resizeWidth, args.imgresize))


img_tmp_dir = "work_dir/test_A"

if __name__ == "__main__":
	# check if cuda is available in your system.
	if torch.cuda.is_available():
	    count_cuda_device = torch.cuda.device_count()
	    for i in range(count_cuda_device):
	        ss=str(i) if i==0 else ss+','+str(i) 
	    os.system("export CUDA_VISIBLE_DEVICES=%s" % ss)
	    print("export CUDA_VISIBLE_DEVICES=%s" % ss)
	else:
	    print("Your system has no CUDA installed. No GPU detected.")
	    sys.exit(0)

 	myparser = ParserAnybaseocr()
 	args = myparser.get_parameters('ocrd-anybaseocr-dewarp')

	files = myparser.parseXML()
	fname=[]

	for i, f in enumerate(files):
		base = str(f).split('.')[0]
		img_dir = os.path.dirname(str(f))
		os.system("mkdir -p %s" % img_tmp_dir)
		os.system("cp %s %s" % (str(f), os.path.join(img_tmp_dir, os.path.basename(str(f)))))
		fname.append(base + '.dw.png')
	dewarping(img_tmp_dir, img_dir)
	os.system("rm -r %s" % img_tmp_dir)
	myparser.write_to_xml(fname, 'DEWARP_')
