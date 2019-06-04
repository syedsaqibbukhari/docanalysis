# ====================================================================
# ====================================
# README file for Binarize component
# ====================================

#Filename : ocrd-anyBaseOCR-binarize.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note:
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** LICENSE ********************
# License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction.
# This method (ocrd-anyBaseOCR-binarize.py) only contains the binarization functionality of ocropus-nlbin.py.
# It still has the same licenses as ocropus-nlbin, i.e Apache 2.0. ((the ocropy license details are pasted below).
# This file is dependend on ocrolib library which comes from https://github.com/tmbdev/ocropy/.

# Copyright 2014 Thomas M. Breuel

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
# limitations under the License.

# *********** LICENSE ********************
# =====================================================================
#!/usr/bin/env python


import argparse
import json
import ocrolib
import os
import os.path
import sys

from pylab import amin, amax, mean, ginput, ones, clip, imshow, median, ion, gray, minimum, array, clf
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
import numpy as np

from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults, print_error
from ..constants import OCRD_TOOL

class OcrdAnybaseocrBinarizer():

    def __init__(self, param):
        self.param = param

    def check_page(self, image):
        if len(image.shape) == 3:
            return "input image is color image %s" % (image.shape,)
        if mean(image) < median(image):
            return "image may be inverted"
        h, w = image.shape
        if h < 600:
            return "image not tall enough for a page image %s" % (image.shape,)
        if h > 10000:
            return "image too tall for a page image %s" % (image.shape,)
        if w < 600:
            return "image too narrow for a page image %s" % (image.shape,)
        if w > 10000:
            return "line too wide for a page image %s" % (image.shape,)
        return None


    def dshow(self, image, info):
        if self.param['debug'] <= 0:
            return
        ion()
        gray()
        imshow(image)
        title(info)
        ginput(1, self.param['debug'])

    def run(self, fname, i):
        print_info("# %s" % (fname))
        print_info("=== %s %-3d" % (fname, i))
        raw = ocrolib.read_image_gray(fname)
        self.dshow(raw, "input")
        # perform image normalization
        image = raw-amin(raw)
        if amax(image) == amin(image):
            print_info("# image is empty: %s" % (fname))
            return
        image /= amax(image)

        if not self.param['nocheck']:
            check = self.check_page(amax(image)-image)
            if check is not None:
                print_error(fname+" SKIPPED. "+check +
                            " (use -n to disable this check)")
                return

        # check whether the image is already effectively binarized
        if self.param['gray']:
            extreme = 0
        else:
            extreme = (np.sum(image < 0.05) + np.sum(image > 0.95)) * 1.0 / np.prod(image.shape)
        if extreme > 0.95:
            comment = "no-normalization"
            flat = image
        else:
            comment = ""
            # if not, we need to flatten it by estimating the local whitelevel
            print_info("flattening")
            m = interpolation.zoom(image, self.param['zoom'])
            m = filters.percentile_filter(m, self.param['perc'], size=(self.param['range'], 2))
            m = filters.percentile_filter(m, self.param['perc'], size=(2, self.param['range']))
            m = interpolation.zoom(m, 1.0/self.param['zoom'])
            if self.param['debug'] > 0:
                clf()
                imshow(m, vmin=0, vmax=1)
                ginput(1, self.param['debug'])
            w, h = minimum(array(image.shape), array(m.shape))
            flat = clip(image[:w, :h]-m[:w, :h]+1, 0, 1)
            if self.param['debug'] > 0:
                clf()
                imshow(flat, vmin=0, vmax=1)
                ginput(1, self.param['debug'])

        # estimate low and high thresholds
        print_info("estimating thresholds")
        d0, d1 = flat.shape
        o0, o1 = int(self.param['bignore']*d0), int(self.param['bignore']*d1)
        est = flat[o0:d0-o0, o1:d1-o1]
        if self.param['escale'] > 0:
            # by default, we use only regions that contain
            # significant variance; this makes the percentile
            # based low and high estimates more reliable
            e = self.param['escale']
            v = est-filters.gaussian_filter(est, e*20.0)
            v = filters.gaussian_filter(v**2, e*20.0)**0.5
            v = (v > 0.3*amax(v))
            v = morphology.binary_dilation(v, structure=ones((int(e*50), 1)))
            v = morphology.binary_dilation(v, structure=ones((1, int(e*50))))
            if self.param['debug'] > 0:
                imshow(v)
                ginput(1, self.param['debug'])
            est = est[v]
        lo = stats.scoreatpercentile(est.ravel(), self.param['lo'])
        hi = stats.scoreatpercentile(est.ravel(), self.param['hi'])
        # rescale the image to get the gray scale image
        print_info("rescaling")
        flat -= lo
        flat /= (hi-lo)
        flat = clip(flat, 0, 1)
        if self.param['debug'] > 0:
            imshow(flat, vmin=0, vmax=1)
            ginput(1, self.param['debug'])
        binarized = 1*(flat > self.param['threshold'])

        # output the normalized grayscale and the thresholded images
        #print_info("%s lo-hi (%.2f %.2f) angle %4.1f %s" % (fname, lo, hi, angle, comment))
        print_info("%s lo-hi (%.2f %.2f) %s" % (fname, lo, hi, comment))
        print_info("writing")
        if self.param['debug'] > 0 or self.param['show']:
            clf()
            gray()
            imshow(binarized)
            ginput(1, max(0.1, self.param['debug']))
        base, _ = ocrolib.allsplitext(fname)
        ocrolib.write_image_binary(base+".bin.png", binarized)
        ocrolib.write_image_gray(base+".nrm.png", flat)
        # print("########### File path : ", base+".nrm.png")
        # write_to_xml(base+".bin.png")
        return base+".bin.png"


def main():
    parser = argparse.ArgumentParser("""
    Image binarization using non-linear processing.

            python ocrd-anyBaseOCR-binarize.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

    This is a compute-intensive binarization method that works on degraded
    and historical book pages.
    """)

    parser.add_argument('-p', '--parameter', type=str, help="Parameter file location")
    parser.add_argument('-w', '--work', type=str, help="Working directory location", default=".")
    parser.add_argument('-I', '--Input', default=None, help="Input directory")
    parser.add_argument('-O', '--Output', default=None, help="output directory")
    parser.add_argument('-m', '--mets', default=None, help="METs input file")
    parser.add_argument('-o', '--OutputMets', default=None, help="METs output file")
    parser.add_argument('-g', '--group', default=None, help="METs image group id")
    args = parser.parse_args()

    # Read parameter values from json file
    param = {}
    if args.parameter:
        with open(args.parameter, 'r') as param_file:
            param = json.loads(param_file.read())
    param = parse_params_with_defaults(param, OCRD_TOOL['tools']['ocrd-anybaseocr-binarize']['parameters'])
    #  print(param)
    # End to read parameters

    # mandatory parameter check
    if not args.mets or not args.Input or not args.Output or not args.work:
        parser.print_help()
        print("Example: ocrd_anyBaseOCR_binarize.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
        sys.exit(0)

    if args.work:
        if not os.path.exists(args.work):
            os.mkdir(args.work)

    binarizer = OcrdAnybaseocrBinarizer(param)
    files = parseXML(args.mets, args.Input)
    fnames = []
    for i, fname in enumerate(files):
        fnames.append(binarizer.run(str(fname), i+1))
    write_to_xml(fnames, args.mets, args.Output, args.OutputMets, args.work)
