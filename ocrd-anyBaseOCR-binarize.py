#!/usr/bin/env python
#====================================================================
#====================================
#README file for Binarize component
#====================================

#Filename : ocrd-anyBaseOCR-binarize.py

#Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
#Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
#Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note: 
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

#*********** LICENSE ********************
# License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction. 
# This method (ocrd-anyBaseOCR-binarize.py) only contains the binarization functionality of ocropus-nlbin.py. 
# It still has the same licenses as ocropus-nlbin, i.e Apache 2.0. ((the ocropy license details are pasted below).
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
#=====================================================================
from __future__ import print_function
from pylab import *
from numpy.ctypeslib import ndpointer
import argparse,os,os.path,glob
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
import multiprocessing
import ocrolib
import json
from pprint import pprint
from xml.dom import minidom
#import xml.etree.ElementTree
#from lxml import etree


parser = argparse.ArgumentParser("""
Image binarization using non-linear processing.
	
	python ocrd-anyBaseOCR-binarize.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

This is a compute-intensive binarization method that works on degraded
and historical book pages.
""")

parser.add_argument('-p','--parameter',type=str,help="Parameter file location")
parser.add_argument('-w','--work',type=str,help="Working directory location", default=".")
parser.add_argument('-I','--Input',default=None,help="Input directory")
parser.add_argument('-O','--Output',default=None,help="output directory")
parser.add_argument('-m','--mets',default=None,help="METs input file")
parser.add_argument('-o','--OutputMets',default=None,help="METs output file")
parser.add_argument('-n','--nocheck',action="store_true", help="disable error checking on inputs")
parser.add_argument('--show', action='store_true', help='display debug result')
parser.add_argument('-gr','--gray',action='store_true',help='force grayscale processing even if image seems binary')
args = parser.parse_args()

def parseXML(fpath):
    input_files=[]
    xmldoc = minidom.parse(fpath)
    nodes = xmldoc.getElementsByTagName('mets:fileGrp')
    for attr in nodes:
        if attr.attributes['USE'].value==args.Input:
            childNodes = attr.getElementsByTagName('mets:FLocat')
            for f in childNodes:
                input_files.append(f.attributes['xlink:href'].value)
    return input_files

def write_to_xml(fpath):
    xmldoc = minidom.parse(args.mets)
    subRoot = xmldoc.createElement('mets:fileGrp')
    subRoot.setAttribute('USE', args.Output)

    for f in fpath:
        #basefile = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        basefile = ocrolib.allsplitext(os.path.basename(f))[0]
        child = xmldoc.createElement('mets:file')
        child.setAttribute('ID', 'BIN_'+basefile)
        child.setAttribute('GROUPID', 'P_' + basefile)
        child.setAttribute('MIMETYPE', "image/png")

        subChild = xmldoc.createElement('mets:FLocat')
        subChild.setAttribute('LOCTYPE', "URL")
        subChild.setAttribute('xlink:href', f)

        #xmldoc.getElementsByTagName('mets:file')[0].appendChild(subChild);
        subRoot.appendChild(child)
        child.appendChild(subChild)

    #subRoot.appendChild(child)
    xmldoc.getElementsByTagName('mets:fileSec')[0].appendChild(subRoot);
    
    if not args.OutputMets:
        metsFileSave = open(os.path.join(args.work, os.path.basename(args.mets)), "w")
    else:
        metsFileSave = open(os.path.join(args.work, args.OutputMets if args.OutputMets.endswith(".xml") else args.OutputMets+'.xml'), "w")
    metsFileSave.write(xmldoc.toxml()) 

def print_info(*objs):
    print("INFO: ", *objs, file=sys.stdout)

def print_error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)

def check_page(image):
    if len(image.shape)==3: return "input image is color image %s"%(image.shape,)
    if mean(image)<median(image): return "image may be inverted"
    h,w = image.shape
    if h<600: return "image not tall enough for a page image %s"%(image.shape,)
    if h>10000: return "image too tall for a page image %s"%(image.shape,)
    if w<600: return "image too narrow for a page image %s"%(image.shape,)
    if w>10000: return "line too wide for a page image %s"%(image.shape,)
    return None

def dshow(image,info):
    if args.debug<=0: return
    ion(); gray(); imshow(image); title(info); ginput(1,args.debug)

def process1(job):
    fname,i = job
    print_info("# %s" % (fname))
    if args.parallel<2: print_info("=== %s %-3d" % (fname, i))
    raw = ocrolib.read_image_gray(fname)
    dshow(raw,"input")
    # perform image normalization
    image = raw-amin(raw)
    if amax(image)==amin(image):
        print_info("# image is empty: %s" % (fname))
        return
    image /= amax(image)

    if not args.nocheck:
        check = check_page(amax(image)-image)
        if check is not None:
            print_error(fname+" SKIPPED. "+check+" (use -n to disable this check)")
            return

    # check whether the image is already effectively binarized
    if args.gray:
        extreme = 0
    else:
        extreme = (sum(image<0.05)+sum(image>0.95))*1.0/prod(image.shape)
    if extreme>0.95:
        comment = "no-normalization"
        flat = image
    else:
        comment = ""
        # if not, we need to flatten it by estimating the local whitelevel
        if args.parallel<2: print_info("flattening")
        m = interpolation.zoom(image,args.zoom)
        m = filters.percentile_filter(m,args.perc,size=(args.range,2))
        m = filters.percentile_filter(m,args.perc,size=(2,args.range))
        m = interpolation.zoom(m,1.0/args.zoom)
        if args.debug>0: clf(); imshow(m,vmin=0,vmax=1); ginput(1,args.debug)
        w,h = minimum(array(image.shape),array(m.shape))
        flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)
        if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)


    # estimate low and high thresholds
    if args.parallel<2: print_info("estimating thresholds")
    d0,d1 = flat.shape
    o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if args.escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = args.escale
        v = est-filters.gaussian_filter(est,e*20.0)
        v = filters.gaussian_filter(v**2,e*20.0)**0.5
        v = (v>0.3*amax(v))
        v = morphology.binary_dilation(v,structure=ones((int(e*50),1)))
        v = morphology.binary_dilation(v,structure=ones((1,int(e*50))))
        if args.debug>0: imshow(v); ginput(1,args.debug)
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),args.lo)
    hi = stats.scoreatpercentile(est.ravel(),args.hi)
    # rescale the image to get the gray scale image
    if args.parallel<2: print_info("rescaling")
    flat -= lo
    flat /= (hi-lo)
    flat = clip(flat,0,1)
    if args.debug>0: imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)
    bin = 1*(flat>args.threshold)

    # output the normalized grayscale and the thresholded images
    #print_info("%s lo-hi (%.2f %.2f) angle %4.1f %s" % (fname, lo, hi, angle, comment))
    print_info("%s lo-hi (%.2f %.2f) %s" % (fname, lo, hi, comment))
    if args.parallel<2: print_info("writing")
    if args.debug>0 or args.show: clf(); gray();imshow(bin); ginput(1,max(0.1,args.debug))
    base,_ = ocrolib.allsplitext(fname)
    ocrolib.write_image_binary(base+".bin.png",bin)
    ocrolib.write_image_gray(base+".nrm.png",flat)
    #print("########### File path : ", base+".nrm.png")
    #write_to_xml(base+".bin.png")
    return base+".bin.png"

def parse_data(arguments):
	arguments = arguments['tools']['ocrd-anyBaseOCR-bin']['parameters']

	for key, val in arguments.items():
		parser.add_argument('--%s' % key,
	            type=eval(val["type"]),
	            help=val["description"],
	            default=val["default"])
	return parser

## Read parameter values from json file
def get_parameters():
    if args.parameter:
        if not os.path.exists(args.parameter):
            print("Error : Parameter file does not exists.")
            sys.exit(0)
        else:
            with open(args.parameter) as json_file:
                json_data = json.load(json_file)
    else:
        parameter_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(os.path.join(parameter_path, 'ocrd-anyBaseOCR-parameter.json')):
            print("Error : Parameter file does not exists.")
            sys.exit(0)
        else:
            with open(os.path.join(parameter_path, 'ocrd-anyBaseOCR-parameter.json')) as json_file:
                json_data = json.load(json_file)
    parser = parse_data(json_data)
    parameters = parser.parse_args()
    return parameters


if __name__ == "__main__":
	args = get_parameters()

	# mendatory parameter check
	if not args.mets or not args.Input or not args.Output or not args.work:
	    parser.print_help()
	    print("Example: python ocrd-anyBaseOCR-binarize.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
	    sys.exit(0)


	if args.debug>0 or args.show: args.parallel = 0

	if args.work:
	    if not os.path.exists(args.work):
	        os.mkdir(args.work)

	files = parseXML(args.mets)
	fname=[]
	for i, f in enumerate(files):
	    fname.append(process1((str(f),i+1)))
	write_to_xml(fname)
