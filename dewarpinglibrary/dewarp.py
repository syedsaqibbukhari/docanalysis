#! /usr/bin/env python
# ========================================================================
# README file for helping function for dewarping component
# ====================================

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note: this work has been done in DFKI, Kaiserslautern, Germany.

# *********** LICENSE ********************
# Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Apache License 2.0

# A permissive license whose main conditions require preservation of copyright 
# and license notices. Contributors provide an express grant of patent rights. 
# Licensed works, modifications, and larger works may be distributed under 
# different terms and without source code.

# *********** LICENSE ********************
# ========================================================================


import sys , os
import numpy
from numpy import *
from numpy import linalg, var, random, take, transpose, dot
from numpy import ones, zeros, array, where, shape , arange, sum
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import copy
import pylab
from pylab import find , plot, matshow, show, arctan, imread
import PIL
import PIL.Image as Image
import PIL.ImageFilter
import Image , ImageDraw
import re
import string
import math
from math import ceil, e, sqrt
from scipy import median
from re import split
import time
from time import clock,time
import scipy
import cv2

def read_image_i(imf):
	I = cv2.imread(imf, 0)
	rows = len(I)
	cols = len(I[0])
	return I,rows,cols												#returns the image pixel values and the rows and columns in image
	
def write_image(Ic,imf):
	im = PIL.Image.new('L',(len(Ic[0]),len(Ic)))
	Ic1 = copy.deepcopy(Ic)
	Ic1.shape = len(Ic)*len(Ic[0])
	im.putdata(Ic1)
	im.save(imf)	

# Histogram Based approach for finding the connected components in the document image

def connected_components(I):
    s=ones((3,3))
    Ij,k = ndimage.label(I,s)
    idj = arange(1,k+1)
    x,y = where(Ij>0)
    A = Ij[x,y]
    args = A.argsort()
    x = x[args]
    y = y[args]
    A = A[args]
    h1,h2 = histogram(A,k)
    xj = []
    yj = []
    msum = 0   
    for i in range(0,len(h1)):
        tsum = msum + h1[i]
        xj.append(x[msum:tsum])
        yj.append(y[msum:tsum])
        msum = msum + h1[i]
    return Ij,idj,xj,yj

def connected_components_statistics(Adjecent_IDs,Adjecnet_x_coordiantes,Adjecnet_y_coordiantes):
	# Find Bounding boxes around smeared adjecent components     
    number_Adj_IDs = len(Adjecent_IDs);
    heights = zeros((number_Adj_IDs),dtype=float); # heighths of adjecent components
    lengths = zeros((number_Adj_IDs),dtype=float); # lengths of adjecent components
    TopX = zeros((number_Adj_IDs),dtype=float);
    TopY = zeros((number_Adj_IDs),dtype=float);
    BotX = zeros((number_Adj_IDs),dtype=float);
    BotY = zeros((number_Adj_IDs),dtype=float);

    for i in range(0,number_Adj_IDs):
        TopX[i] = array(Adjecnet_x_coordiantes[i]).min();
        TopY[i] = array(Adjecnet_y_coordiantes[i]).min();
        BotX[i] = array(Adjecnet_x_coordiantes[i]).max()
        BotY[i] = array(Adjecnet_y_coordiantes[i]).max();
        heights[i] = BotX[i] - TopX[i] + 1;
        lengths[i] = BotY[i] - TopY[i] + 1;
    return lengths,heights,TopX,TopY,BotX,BotY;
    
def doc_cleanup(imf):
	[Ic,rows,columns] = read_image_i(imf)
	
	# Threshold value set for removing the additional pixels
	I = copy.deepcopy(Ic)
	I = 1 - (Ic/255)
	# Creates label of the Image 
	[imj,ids,xj,yj] = connected_components(I)
		
	# Calculates the length and height of the Connected Components [calculate xy-coordinate of each rect and their height & width]
	[li,hi,txi,tyi,bxi,byi]=connected_components_statistics(ids,xj,yj);

		
	hia = hi[where(hi>1)[0]]
	lia = li[where(li>1)[0]]
		
	# Threshold for big Components
	H_thr = (rows/10)
	L_thr = (columns/10)
		
	#Threshold for small components
	H_sdd = int( mean(hia) + 20*sqrt(var(hia)) )
	L_sdd = int( mean(lia) + 20*sqrt(var(lia)) )
		
	Iclean = copy.deepcopy(Ic)
		
	for i in range(0,len(xj)):
		if hi[i] > H_sdd or li[i] > L_sdd:
			Iclean[xj[i],yj[i]] = 255
	
	write_image(Iclean,imf)
	
def ridges_thinning_using_erosion(Isn):
	import mmorph
	se = ones((2,1),dtype=int32)
	Isn = array(Isn,dtype=int32)
	Isne = mmorph.erode(Isn,se)
	Isne = Isne+1
	return Isne;
  
def read_ridges_thinning_erosion(isnakes,Ic_,rows,columns):
	[aisnakes,snakesIDs,sr,sc] = connected_components(isnakes)			#sr and sc are the x and y set of coordinates of CC's in ridges
	s = [];
	Thr = 0.1*columns
	for i in range(0,len(snakesIDs)):
		sid = argsort(sc[i])
		sr1 = sr[i][sid]
		sc1 = sc[i][sid]
		if not((sc1[0]<25 or sc1[-1]>columns-25) and len(sc1) < Thr):
			j=0;
			while Ic_[sr1[j],sc1[j]]==0 and j < len(sc1)-1:
				j=j+1;
			start = j;
			j = len(sc1)-1
			while Ic_[sr1[j],sc1[j]]==0 and j>0:
				j=j-1;
			end = j;
			if start<=end:
				s1 = []
				s1.append(copy.deepcopy(sr1[start:end+1]))
				s1.append(copy.deepcopy(sc1[start:end+1]))
				s.append(s1)
	return s;

def textline_finding(imf):
	[df,ex]=os.path.splitext(imf)												# splits the format of the file name
	imfi = df + "_invert.pgm"												# Changes the format of file name to .pgm
	os.system("convert "+imf+" -negate -depth 8 "+imfi)									# Converts the file into pgm format using system call
	
	[im,rows,columns] = read_image_i(imfi)

	[df,ex]=os.path.splitext(imfi)
	ssimf = df + "_s.pgm"
	
	os.system("cp "+imfi+" "+ssimf)
	os.system("./dewarpinglibrary/textlinefinder "+imfi+" "+ssimf) 						# For finding out the ridges inside a document
	
	Isn, r,c = read_image_i(ssimf)
	sn = read_ridges_thinning_erosion(Isn,im,rows,columns)

	os.system("rm "+imfi)
	os.system("rm "+ssimf)
	return im/255,rows,columns,sn;
	
def update_sdb_correct(imj,ids,sdb2):
	
	isn = []
	allk = {}
	allkl = {}	
	
	for i in range(0,len(sdb2)):
		k = imj[sdb2[i][0][0],sdb2[i][1][0]]
		if allk.has_key(k):
			allk[k].append(i)
			allkl[k].append(len(sdb2[i][0]))
		else:
			allk[k] = [i]
			allkl[k] = [len(sdb2[i][0])]
	
	akeys = allk.keys()
	
	for i in akeys:
		isn.append(sdb2[allk[i][argmax(allkl[i])]])
	
	return isn
	
	
def small_connected_components(imj,ids,xj,yj,sdb2,r,c):
	
	idsc = list(copy.deepcopy(ids))									# Contains all the connected component's
	ids_ = []
	imj_ = copy.deepcopy(imj)
	
	for i in range(0,len(sdb2)):
		k = imj[sdb2[i][0][0],sdb2[i][1][0]]
		idsc.remove(k)										# Removes the longest connected component's from idsc
		ids_.append(k)										# Fills ids_ with all the long connected component's
	
	for i in range(0,len(idsc)):
		imj_[xj[idsc[i]-1],yj[idsc[i]-1]] = 0							# Fills the pixel value in the image as zero for the small connected component's
	
	for i in range(0,len(idsc)):
		
		x1 = min(xj[idsc[i]-1])-20								# Creates a rectangular region around the small ridge to find the ridge it originally belongs to
		x2 = max(xj[idsc[i]-1])+20
		y1 = min(yj[idsc[i]-1])-20
		y2 = max(yj[idsc[i]-1])+20
		
		if x1<0:										# Manipulation's needed if the boundary of rectangular region goes outside the area of interest
			x1 = 0
		if x2>r-1:
			x2 = r-1
		if y1<0:
			y1=0
		if y2>c-1:
			y2 = c-1
		
		t = imj_[x1:x2,y1:y2]
		tk = t[where(t>0)]
		
		if len(tk)>0:
			imj_[xj[idsc[i]-1],yj[idsc[i]-1]] = tk[0]		
	
	return imj_,ids_,idsc


def textline_labeling(im,sdb,r,c):
	
	I = copy.deepcopy(im)
	
	lines = I*0
	xlines = []
	ylines = []
	idslines = []
	xremain = []
	yremain = []
	idsremain = []
	
	for i in range(0,len(sdb)):															# Used for thickning of the lines so that it may be able to cover all the ridges of a particular line
		for j in range(0,5):
			I[sdb[i][0]-j,sdb[i][1]] = 1
	
	[imj,ids,xj,yj] = connected_components(I)
	
	sdb = update_sdb_correct(imj,ids,sdb)												# Updates the ridges of only those connected components which are longest in the length
	
	imj_, ids_,idsc = small_connected_components(imj,ids,xj,yj,sdb,r,c) 				# Separates the small CC's type ridges present in document with the main textlies and updates them
	
	imj2 = imj_*im
	imj2 = imj2.transpose()
	xj2 = []
	yj2 = []
	ids2 = list(copy.deepcopy(ids_))
	dict = {}
	
	for i in range(0,len(ids2)):
		dict[ids2[i]] = i+1
	
	for i in range(0,len(ids2)):
		xj2.append([])
		yj2.append([])
	x,y = where(imj2>0)
	
	for i in range(0,len(x)):
		a = imj2[x[i]][y[i]]
		xj2[dict[a]-1].append(y[i])
		yj2[dict[a]-1].append(x[i])
	
	for i in range(0,len(xj2)):
		xj2[i] = array(xj2[i])
		yj2[i] = array(yj2[i])

	imj2 = imj2.transpose()
	k = []
	
	for i in range(0,len(xj2)):
		if len(xj2[i])==0:
			k.append(i)
	kc = 0
	
	for i in range(0,len(k)):
		t = sdb.pop(k[i]-kc)
		t = xj2.pop(k[i]-kc)
		t = yj2.pop(k[i]-kc)
		kc = kc + 1
	
	return imj2,xj2,yj2,sdb
	
def combination(s1,s2,diff):
	snc = []	
	sx = []
	sy = []
	mini = []
	maxi = []
	sx.append(s1[0])
	sy.append(s1[1])
	mini.append(min(s1[1]))
	maxi.append(max(s1[1]))
	sx.append(s2[0]+diff)
	sy.append(s2[1])
	mini.append(min(s2[1]))
	maxi.append(max(s2[1]))
	minval = min(mini)
	maxval = max(maxi)
	y = array(range(minval,maxval+1))
	x = y*0
	for j in range(0,len(y)):
		tx = 0
		count = 0
		for k in range(0,len(sx)):
			id1 = where(sy[k]==y[j])[0]
			tx = tx + sum(sx[k][id1])
			count = count + len(id1)
		x[j] = tx/count
	snc.append(x)
	snc.append(y)
	
	return snc

def row_number_estimation_for_dewarped_lines(s,rows,cols):
	
	a =300
	I = array(zeros((rows,cols)))
	for i in range(0,len(s)):
		I[s[i][0],s[i][1]]=i+1
	Allids = []
	snakeid = []
	snakedist = []
	snakelength = []
	dewarpedrowid = []
	for i in range(0,len(s)):
		m = int(len(s[i][0])/2)
		p = []
		p.append([s[i][0][0],s[i][1][0]])
		p.append([s[i][0][m],s[i][1][m]])
		p.append([s[i][0][-1],s[i][1][-1]])
		for j in range(1,2):
			rs = p[j][0]-a
			re = p[j][0]+a
			if rs<0:
				rs=0
			if rs>rows-1:
				rs=rows-1
			if re<0:
				re=0
			if re>rows-1:
				re=rows-1
			It1 = I[range(rs,re),p[j][1]]
			dit = where(It1>0)[0]
			if len(dit)>0:
				for t in range(0,len(dit)):
					sid = int(It1[dit[t]]-1)
					snakeid.append(sid)
					dist = p[j][0] - where(I[:,p[j][1]]== sid+1)[0][0]
					snakedist.append(dist)
					snakelength.append(len(s[sid][0]))
		
		k = where(array(snakelength)==max(snakelength))[0][0]
		ss = combination(s[i],s[snakeid[k]],snakedist[k])
		val = int(average(ss[0]))
		dewarpedrowid.append(val)
		snakeid = []
		snakedist = []	
		snakelength = []
	return dewarpedrowid


def geometric_distortion_correction(sdb2,lines,xlines,ylines,rows,cols):
	
	dwprows = row_number_estimation_for_dewarped_lines(sdb2,rows,cols)									# Row number Estimation for removing the Geometric Distortion
	
	I = zeros((rows,cols))
	
	for i in range(0,len(sdb2)):
		diff = dwprows[i] - sdb2[i][0]
		dict = {}
		for j in range(0,len(sdb2[i][0])):
			dict[sdb2[i][1][j]] = diff[j]
		first = dict[dict.keys()[0]]
		last = dict[dict.keys()[-1]]
		keystart = 0;
		for j in range(0,len(xlines[i])):
			if dict.has_key(ylines[i][j]):
				keystart = 1
				xlines[i][j] = xlines[i][j] + dict[ylines[i][j]] 
			else:
				if keystart==0:
					xlines[i][j] = xlines[i][j] + first
				else:		
					xlines[i][j] = xlines[i][j] + last
		xlines[i][where(xlines[i]>rows-1)] = rows-1
		ylines[i][where(ylines[i]>cols-1)] = cols-1
		I[xlines[i],ylines[i]] = 1
	
	return I
	
	
def small_textline_filtering(s):
	num = len(s)
	avglen = 0.0;
	alllen = []
	for i in range(0,num):
		alllen.append(len(s[i][0]))
	alllen = array(alllen)
	avg = sum(alllen)/len(alllen)
	avg = avg/2
	maxper = 0.2*max(alllen)
	sn = []
	for i in range(0,num):
		if alllen[i] > maxper and alllen[i]>=avg:
			sn.append(copy.deepcopy(s[i]))
	return sn
	
def choice(seq, n = 1):
    indx = arange(len(seq))
    random.shuffle(indx)
    return take(seq, indx[0:n],axis=0)
    
def RANSAC(data,n,k,t,d):

    model_params = array([0.0,0.0]) # slope and intercept
    model_error = 1e6
    model_inliers = []

    for i in range(k): # k iterations
        inliers = choice(data,n)
        a = transpose(array([inliers[:,0], ones(n)]))
        b = inliers[:,1]
        (p, residuals, rank, s) = linalg.lstsq(a,b)
        compatible = []
        for pt in data:
            if abs(pt[1] - dot(p, [pt[0], 1.0])) < t:
                compatible.append(pt)
        
        if len(compatible) > d:
            # The current model is good enough so we should recompute it using all compatible points.
            compatible = array(compatible)
            a = transpose(array([compatible[:,0], ones(len(compatible))]))
            b = compatible[:,1]
            (p, residuals, rank, s) = linalg.lstsq(a,b)
            
            if residuals < model_error:
                model_params = p
                model_error = residuals
                model_inliers = compatible

    return (model_params, model_error, model_inliers)
    
def ransac_repeat(data,n,k,t):
	model_inliers = []
	D = len(data)
	while D>2 and len(model_inliers)==0:
		t = t + 0.2
		model_params, model_error, model_inliers = RANSAC(data,n,k,t,D)
		D=D-4;
	return model_inliers
	
def border_estimation_ransac(s,rows,cols,mls):
	do2nd = 1
	xl = []
	yl = []
	xr = []
	yr = []
	for i in range(0,len(s)):
		xl.append(s[i][0][0])
		yl.append(s[i][1][0])
		xr.append(s[i][0][-1])
		yr.append(s[i][1][-1])
	lla = []
	lla.append(array(xl))
	lla.append(array(yl))
	
	lra = []
	lra.append(array(xr))
	lra.append(array(yr))
	
	lla = array(lla)
	lra = array(lra)
	lla = lla.transpose()
	lra = lra.transpose()
	
	ll = ransac_repeat(lla,2,50,mls)
	lr = ransac_repeat(lra,2,50,mls)
	
	if len(ll)>0 and len(lr)>0:
		sm = pylab.polyfit(ll[:,0],ll[:,1],1)
		xx = arange(0,rows)
		yy = xx*sm[0] + sm[1]
		ll = []
		ll.append(copy.deepcopy(xx))
		ll.append(copy.deepcopy(yy))
		
		sm = pylab.polyfit(lr[:,0],lr[:,1],1)
		xx = arange(0,rows)
		yy = xx*sm[0] + sm[1]
		lr = []
		lr.append(copy.deepcopy(xx))
		lr.append(copy.deepcopy(yy))
	else:
		do2nd = 0
	return ll,lr,do2nd
	
def left_right_borders_checking(ll,lr,sdb2):
	topdist = lr[1][0]-ll[1][0]
	botdist = lr[1][-1]-ll[1][-1]
	al = []
	for i in range(0,len(sdb2)):
		al.append(len(sdb2[i][0]))
		
	max1 = max(al)
	al.remove(max1)
	max3 = max1
	try:
		max2 = max(al)
		al.remove(max2)
	except:
		max2 = max1
	try:
		max3 = max(al)
		al.remove(max3)
	except:
		max3 = max2
	maxa = (max1+max2+max3)/3.0
	return max(topdist,botdist)/maxa
	
def homography_perspective_rectification_estimation(ll,lr,inv):
	x1_ = ll[0][0]
	y1_ = ll[1][0]
	x2_ = lr[0][0]
	y2_ = lr[1][0]
	
	x3_ = ll[0][-1]
	y3_ = ll[1][-1]
	x4_ = lr[0][-1]
	y4_ = lr[1][-1]
	
	if (y4_-y3_)>=(y2_-y1_):
		ys = y3_
		ye = y4_ 
	else:
		ys = y1_ 
		ye = y2_
		
	x1 = ll[0][0]
	y1 = ys
	x2 = lr[0][0]
	y2 = ye
	
	x3 = lr[0][-1]
	y3 = ys 
	x4 = ll[0][-1]
	y4 = ye
	
	ll_ = []
	x = []
	y = []
	x.append(x1_)
	x.append(x3_)
	y.append(y1_)
	y.append(y3_)
	ll_.append(x)
	ll_.append(y)
	
	lr_ = []
	x = []
	y = []
	x.append(x2_)
	x.append(x4_)
	y.append(y2_)
	y.append(y4_)
	lr_.append(x)
	lr_.append(y)
	
	A = array(zeros((8,8)))
	A[0][0] = x1
	A[0][1] = y1
	A[0][2] = 1
	A[0][6] = -x1_*x1
	A[0][7] = -x1_*y1
	
	A[1][3] = x1
	A[1][4] = y1
	A[1][5] = 1
	A[1][6] = -y1_*x1
	A[1][7] = -y1_*y1
	
	A[2][0] = x2
	A[2][1] = y2
	A[2][2] = 1
	A[2][6] = -x2_*x2
	A[2][7] = -x2_*y2
	
	A[3][3] = x2
	A[3][4] = y2
	A[3][5] = 1
	A[3][6] = -y2_*x2
	A[3][7] = -y2_*y2
	
	A[4][0] = x3
	A[4][1] = y3
	A[4][2] = 1
	A[4][6] = -x3_*x3
	A[4][7] = -x3_*y3
	
	A[5][3] = x3
	A[5][4] = y3
	A[5][5] = 1
	A[5][6] = -y3_*x3
	A[5][7] = -y3_*y3
	
	A[6][0] = x4
	A[6][1] = y4
	A[6][2] = 1
	A[6][6] = -x4_*x4
	A[6][7] = -x4_*y4
	
	A[7][3] = x4
	A[7][4] = y4
	A[7][5] = 1
	A[7][6] = -y4_*x4
	A[7][7] = -y4_*y4
	
	R = array(zeros((8)))
	R[0] = x1_
	R[1] = y1_
	R[2] = x2_
	R[3] = y2_
	R[4] = x3_
	R[5] = y3_
	R[6] = x4_
	R[7] = y4_
	
	H = array(ones((9)))
	
	H[0:8] = dot(inv(A),R)
	
	H.shape = 3,3
	
	return H
	
def bilinear_approximation(Ide,x,y,rows,columns):
	
	x[where(x>rows-1)[0]] = rows-1
	x[where(x<0)[0]] = 0
	y[where(y>columns-1)[0]] = columns-1
	y[where(y<0)[0]] = 0
	
	x1 = array(numpy.floor(x),dtype=int)
	x2 = array(numpy.ceil(x),dtype=int)
	y1 = array(numpy.floor(y),dtype=int)
	y2 = array(numpy.ceil(y),dtype=int)
	
	t = x2-x1
	x1 = x1*t
	t = y2-y1
	y1 = y1*t
	X2 = x2-x
	X1 = x-x1
	r1 = ((X2)*Ide[x1,y1] + (X1)*Ide[x2,y1])
	r2 = ((X2)*Ide[x1,y2] + (X1)*Ide[x2,y2])
	val = ((y2-y)*r1+(y-y1)*r2)/(y2-y1)*(x2-x1)
	return val
	
	
def perspective_correction(H,Id,rows,columns,ll,lr):
	Idt = 1-Id		
	endcol = int(max(max(ll[1]),max(lr[1])))
	cols2 = max(columns,endcol)
	cr = range(0,cols2)
	rr = ones((cols2))
	ir = []
	jr = []
	jr = cr*rows
	ijr = ones((rows*cols2))
	for i in range(0,rows):									
		ir.extend(rr*i)
	p_ = dot(H,[ir,jr,ijr])									
	p_[0] = p_[0]/p_[2]
	p_[1] = p_[1]/p_[2]
	Id2 = bilinear_approximation(Idt,p_[0],p_[1],rows,columns)							# Function used to billiner approximate the image pixels after convoluing the image with Homography matrix
	Id2.shape = rows,cols2
	return Id2
	
def perspective_distortion_correction(Id,rows,columns,sdb2):
	sdb2_ = small_textline_filtering(sdb2)												# Function used to find the small textlines in the image and remove them from the image to obtain a proper borders
	ll,lr,do2nd = border_estimation_ransac(sdb2_,rows,columns,1)						# Function used to estimate the borders of the image using the image obtained from above with the use of big lines
	if do2nd==1:
		p = left_right_borders_checking(ll,lr,sdb2_)*100	
		if p>=75:
			H = homography_perspective_rectification_estimation(ll,lr,pylab.inv)		# Function used to obtain the Homography Matrix 
			Id2 = perspective_correction(H,Id,rows,columns,ll,lr)						# Function to remove the perspective distortion by multiplying the image pixels by hpmography matrix
			del H
		else:
			Id2 = 1-Id
	else:
		Id2 = 1-Id
	return Id2