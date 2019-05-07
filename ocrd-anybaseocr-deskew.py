#!/usr/bin/env python
#======================================================================
# ====================================
# README file for Skew Correction component
# ====================================

# Filename : ocrd-anybaseocr-deskew.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note: 
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anybaseocr-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and do the skew correction of that document.
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction. 
# This method (ocrd-anybaseocr-deskew.py) only contains the skew correction functionality of ocropus-nlbin.py. 
# It still has the same licenses as ocropus-nlbin, i.e Apache 2.0 (the ocropy license details are pasted below).
# This file is dependend on ocrolib library which comes from https://github.com/tmbdev/ocropy/. 

#Copyright 2014 Thomas M. Breuel

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
# limitations under the License.

#*********** LICENSE ********************
#======================================================================
import argparse
import os
import os.path
import sys
import json

from pylab import amin, amax, linspace, mean, var, plot, ginput, ones, clip, imshow
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
import ocrolib
#from utils import parseXML, write_to_xml, print_info
from ParserAnybaseocr import *

class OcrdAnybaseocrDeskewer():
	def __init__(self, args):
		self.args = args

	def estimate_skew_angle(self, image, angles):
		args = self.args
		estimates = []

		for a in angles:
			v = mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
			v = var(v)
			estimates.append((v, a))
		if args.debug > 0:
			plot([y for x, y in estimates], [x for x, y in estimates])
			ginput(1, args.debug)
		_, a = max(estimates)
		return a

	def deskew(self, fpath, job):
		args = self.args
		base, _ = ocrolib.allsplitext(fpath)
		basefile = ocrolib.allsplitext(os.path.basename(fpath))[0]
		if args.parallel < 2:
			myparser.print_info("=== %s %-3d" % (fpath, job))
		raw = ocrolib.read_image_gray(fpath)
		flat = raw
		# estimate skew angle and rotate
		if args.maxskew > 0:
			if args.parallel < 2:
				myparser.print_info("estimating skew angle")
			d0, d1 = flat.shape
			o0, o1 = int(args.bignore*d0), int(args.bignore*d1)
			flat = amax(flat)-flat
			flat -= amin(flat)
			est = flat[o0:d0-o0, o1:d1-o1]
			ma = args.maxskew
			ms = int(2*args.maxskew*args.skewsteps)
			angle = self.estimate_skew_angle(est, linspace(-ma, ma, ms+1))
			flat = interpolation.rotate(flat, angle, mode='constant', reshape=0)
			flat = amax(flat)-flat
		else:
			angle = 0

		# estimate low and high thresholds
		if args.parallel < 2:
			myparser.print_info("estimating thresholds")
		d0, d1 = flat.shape
		o0, o1 = int(args.bignore*d0), int(args.bignore*d1)
		est = flat[o0:d0-o0, o1:d1-o1]
		if args.escale > 0:
			# by default, we use only regions that contain
			# significant variance; this makes the percentile
			# based low and high estimates more reliable
			e = args.escale
			v = est-filters.gaussian_filter(est, e*20.0)
			v = filters.gaussian_filter(v**2, e*20.0)**0.5
			v = (v > 0.3*amax(v))
			v = morphology.binary_dilation(v, structure=ones((int(e*50), 1)))
			v = morphology.binary_dilation(v, structure=ones((1, int(e*50))))
			if args.debug > 0:
				imshow(v)
				ginput(1, args.debug)
			est = est[v]
		lo = stats.scoreatpercentile(est.ravel(), args.lo)
		hi = stats.scoreatpercentile(est.ravel(), args.hi)
		# rescale the image to get the gray scale image
		if args.parallel < 2:
			myparser.print_info("rescaling")
		flat -= lo
		flat /= (hi-lo)
		flat = clip(flat, 0, 1)
		if args.debug > 0:
			imshow(flat, vmin=0, vmax=1)
			ginput(1, args.debug)
		deskewed = 1*(flat > args.threshold)

		# output the normalized grayscale and the thresholded images
		myparser.print_info("%s lo-hi (%.2f %.2f) angle %4.1f" % (basefile, lo, hi, angle))
		if args.parallel < 2:
			myparser.print_info("writing")
		ocrolib.write_image_binary(base+".ds.png", deskewed)
	    	return base+".ds.png"			

if __name__ == "__main__":
 	myparser = ParserAnybaseocr()
 	args = myparser.get_parameters('ocrd-anybaseocr-deskew')

	deskewer = OcrdAnybaseocrDeskewer(args)
	files = myparser.parseXML()
	fname = []
	for i, f in enumerate(files):
		fname.append(deskewer.deskew(str(f), i+1))

	myparser.write_to_xml(fname, 'DESKEW_')
