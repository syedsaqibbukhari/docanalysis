# ========================================================================
# README file for Dewarping component
# ====================================

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note: 
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) At the moment there are no exposed parameters that can be changed.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (work_dir/mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

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
from numpy import linalg, var, random, take, transpose, dot
from numpy import ones, zeros, array, where, shape, arange, sum
from scipy.ndimage import gaussian_filter1d
import copy
import pylab
from pylab import find , plot, matshow, show, arctan
import PIL
import PIL.Image
import PIL.ImageFilter
import Image , ImageDraw
import re
import string
import math
from math import ceil, e, sqrt
from scipy import median
from re import split
from dewarpinglibrary.dewarp import *
import time
from time import clock,time
import cv2
import ocrolib
import json
from xml.dom import minidom

parser = argparse.ArgumentParser("""
Image pageframe using non-linear processing.
    
    python ocrd-anyBaseOCR-dewarp.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

""")

parser.add_argument('-p','--parameter',type=str,help="Parameter file location")
#parser.add_argument('files',nargs='+')
parser.add_argument('-O','--Output',default=None,help="output directory")
parser.add_argument('-w','--work',type=str,help="Working directory location", default=".")
parser.add_argument('-I','--Input',default=None,help="Input directory")
parser.add_argument('-m','--mets',default=None,help="METs input file")
parser.add_argument('-o','--OutputMets',default=None,help="METs output file")
parser.add_argument('-g','--group',default=None,help="METs image group id")

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
        basefile = ocrolib.allsplitext(os.path.basename(f))[0]
        child = xmldoc.createElement('mets:file')
        child.setAttribute('ID', 'DEWARP_'+basefile)
        child.setAttribute('GROUPID', 'P_' + basefile)
        child.setAttribute('MIMETYPE', "image/png")

        subChild = xmldoc.createElement('mets:FLocat')
        subChild.setAttribute('LOCTYPE', "URL")
        subChild.setAttribute('xlink:href', f)

        subRoot.appendChild(child)
        child.appendChild(subChild)

    #subRoot.appendChild(child)
    xmldoc.getElementsByTagName('mets:fileSec')[0].appendChild(subRoot);

    if not args.OutputMets:
        metsFileSave = open(os.path.join(args.work, os.path.basename(args.mets)), "w")
    else:
        metsFileSave = open(os.path.join(args.work, args.OutputMets if args.OutputMets.endswith(".xml") else args.OutputMets+'.xml'), "w") 
    metsFileSave.write(xmldoc.toxml()) 

def dewarping(imforig):
	[df,ex1]=os.path.splitext(imforig)
	[df,ex2]=os.path.splitext(df)
	imf = df + "_binarized.pgm"
	os.system("cp "+imforig+" "+imf)
	imout = df + "_dewarped.pgm"
	imout2 = df + ".dw"+ex1
	print ""
	print "DEWARPING:  Clean-up   Text-Line-Finding    Removing-Geometric-Distortions    Removing-Perspective-Distortions"
	print ""
	

	print "1/4 : Document Clean-Up"
	doc_cleanup(imf)
	##################################### SNAKE DEFORMATION ######################################
	print "2/4 : Text-Line Finding"
	im,rows,columns,sdb2 = textline_finding(imf)
	s = copy.deepcopy(sdb2)
	############## LINE FINDING W.R.T SNAKES ###############################
	
	lines,xlines,ylines,sdb2 = textline_labeling(im,sdb2,rows,columns)

	del im
	##################################  DEWARPING - GEOMETRIC DISTORTION  #######################
	print "3/4 : Removing Geometric Distortions"
	Id = geometric_distortion_correction(sdb2,lines,copy.deepcopy(xlines),copy.deepcopy(ylines),rows,columns)

	write_image(Id*255,imout)
	del xlines,ylines
	
	################################### DEWARPING - PERSPECTIVE DISTO|  ##########################
	print "4/4 : Removing Perspective Distortions"
	Id2 = perspective_distortion_correction(Id,rows,columns,sdb2)
	del Id,sdb2,s
	#return Id2
	##############################################################################################
	print ""
	print "..... writing ouput as ", imout
	print ""
	write_image(Id2*255,imout)
	os.system("convert "+imout+" -negate -morphology Dilate Disk:0.5 -negate "+imout2)
	os.system("rm "+imf)
	os.system("rm "+imout)

# mendatory parameter check
if not args.mets or not args.Input or not args.Output or not args.work:
    parser.print_help()
    print("Example: python ocrd-anyBaseOCR-dewarp.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
    sys.exit(0)

if args.work:
    if not os.path.exists(args.work):
        os.mkdir(args.work)

files = parseXML(args.mets)
fname=[]

if __name__ == "__main__":
	for i, f in enumerate(files):
		print "Process file: ", str(f) , i+1
		base,_ = ocrolib.allsplitext(str(f))
		dewarping(f)
		fname.append(base + '.dw.png')

	write_to_xml(fname)