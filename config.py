#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

SRC_DIRS = ['./img/iso3200/', './img/iso800/']
DST_DIRS = ['./img/sample3200/', './img/sample800/']
MEAN_IMG_FNAME = 'mean.jpg'
IMREAD_FLAG = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH
IMSHAPE = (3456, 5184)
SHAPE = (300, 300)
OFFSET = tuple((np.array(IMSHAPE) - np.array(SHAPE))/2)
TRANS_PX = 2
BINS = None
SAMPLE_FLG = True
MEAN_FLG = True

PLT_ALPHA = 0.3
PLT_COLORS = ['red', 'blue']
PLT_LABELS = ['ISO 3200, SS 1/4s', 'ISO 800, SS 1s']
