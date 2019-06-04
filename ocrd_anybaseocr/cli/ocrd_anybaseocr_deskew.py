# ======================================================================
# ====================================
# README file for Skew Correction component
# ====================================

# Filename : ocrd-anyBaseOCR-deskew.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note:
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and do the skew correction of that document.
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction.
# This method (ocrd-anyBaseOCR-deskew.py) only contains the skew correction functionality of ocropus-nlbin.py.
# It still has the same licenses as ocropus-nlbin, i.e Apache 2.0 (the ocropy license details are pasted below).
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
# ======================================================================
#!/usr/bin/env python


import argparse
import os
import os.path
import sys
import json

from pylab import amin, amax, linspace, mean, var, plot, ginput, ones, clip, imshow
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
import ocrolib
from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults
from ..constants import OCRD_TOOL

class OcrdAnybaseocrDeskewer():

    def __init__(self, param):
        self.param = param

    def estimate_skew_angle(self, image, angles):
        param = self.param
        estimates = []

        for a in angles:
            v = mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
            v = var(v)
            estimates.append((v, a))
        if param['debug'] > 0:
            plot([y for x, y in estimates], [x for x, y in estimates])
            ginput(1, param['debug'])
        _, a = max(estimates)
        return a


    def run(self, fpath, job):
        param = self.param
        base, _ = ocrolib.allsplitext(fpath)
        basefile = ocrolib.allsplitext(os.path.basename(fpath))[0]

        if param['parallel'] < 2:
            print_info("=== %s %-3d" % (fpath, job))
        raw = ocrolib.read_image_gray(fpath)

        flat = raw
        # estimate skew angle and rotate
        if param['maxskew'] > 0:
            if param['parallel'] < 2:
                print_info("estimating skew angle")
            d0, d1 = flat.shape
            o0, o1 = int(param['bignore']*d0), int(param['bignore']*d1)
            flat = amax(flat)-flat
            flat -= amin(flat)
            est = flat[o0:d0-o0, o1:d1-o1]
            ma = param['maxskew']
            ms = int(2*param['maxskew']*param['skewsteps'])
            angle = self.estimate_skew_angle(est, linspace(-ma, ma, ms+1))
            flat = interpolation.rotate(flat, angle, mode='constant', reshape=0)
            flat = amax(flat)-flat
        else:
            angle = 0

        # estimate low and high thresholds
        if param['parallel'] < 2:
            print_info("estimating thresholds")
        d0, d1 = flat.shape
        o0, o1 = int(param['bignore']*d0), int(param['bignore']*d1)
        est = flat[o0:d0-o0, o1:d1-o1]
        if param['escale'] > 0:
            # by default, we use only regions that contain
            # significant variance; this makes the percentile
            # based low and high estimates more reliable
            e = param['escale']
            v = est-filters.gaussian_filter(est, e*20.0)
            v = filters.gaussian_filter(v**2, e*20.0)**0.5
            v = (v > 0.3*amax(v))
            v = morphology.binary_dilation(v, structure=ones((int(e*50), 1)))
            v = morphology.binary_dilation(v, structure=ones((1, int(e*50))))
            if param['debug'] > 0:
                imshow(v)
                ginput(1, param['debug'])
            est = est[v]
        lo = stats.scoreatpercentile(est.ravel(), param['lo'])
        hi = stats.scoreatpercentile(est.ravel(), param['hi'])
        # rescale the image to get the gray scale image
        if param['parallel'] < 2:
            print_info("rescaling")
        flat -= lo
        flat /= (hi-lo)
        flat = clip(flat, 0, 1)
        if param['debug'] > 0:
            imshow(flat, vmin=0, vmax=1)
            ginput(1, param['debug'])
        deskewed = 1*(flat > param['threshold'])

        # output the normalized grayscale and the thresholded images
        print_info("%s lo-hi (%.2f %.2f) angle %4.1f" % (basefile, lo, hi, angle))
        if param['parallel'] < 2:
            print_info("writing")
        ocrolib.write_image_binary(base+".ds.png", deskewed)
        return base+".ds.png"


def main():
    parser = argparse.ArgumentParser("""
    Image deskewing using non-linear processing.

        python ocrd-anyBaseOCR-deskew.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

    This is a compute-intensive deskew method that works on degraded and historical book pages.
    """)

    parser.add_argument('-p', '--parameter', type=str, help="Parameter file location")
    parser.add_argument('-O', '--Output', default=None, help="output directory")
    parser.add_argument('-w', '--work', type=str, help="Working directory location", default=".")
    parser.add_argument('-I', '--Input', default=None, help="Input directory")
    parser.add_argument('-m', '--mets', default=None, help="METs input file")
    parser.add_argument('-o', '--OutputMets', default=None, help="METs output file")
    parser.add_argument('-g', '--group', default=None, help="METs image group id")

    args = parser.parse_args()

    #args.files = ocrolib.glob_all(args.files)

    # Read parameter values from json file
    param = {}
    if args.parameter:
        with open(args.parameter, 'r') as param_file:
            param = json.loads(param_file.read())
    param = parse_params_with_defaults(param, OCRD_TOOL['tools']['ocrd-anybaseocr-deskew']['parameters'])
    print("%s" % param)
    # End to read parameters

    # mendatory parameter check
    if not args.mets or not args.Input or not args.Output or not args.work:
        parser.print_help()
        print("Example: ocrd-anybaseocr-deskew -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
        sys.exit(0)

    if args.work:
        if not os.path.exists(args.work):
            os.mkdir(args.work)

    deskewer = OcrdAnybaseocrDeskewer(param)
    files = parseXML(args.mets, args.Input)
    fnames = []
    for i, fname in enumerate(files):
        fnames.append(deskewer.run(str(fname), i+1))
    write_to_xml(fnames, args.mets, args.Output, args.OutputMets, args.work)
