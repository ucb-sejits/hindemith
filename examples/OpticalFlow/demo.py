import sys
from numpy import *
import scipy.sparse
import scipy.sparse.linalg
import cv2
import cv2.cv as cv

import middlebury
import flow_mod

params = {}
params['solver_type'] = 'HSJACOBI'
params['num_iterations'] = 100
params['num_pyramid'] = 2
params['num_full'] =1
params['alpha'] = 0.1

# Instantiate Optical Flow object
import hs_solver_jacobi
OpticalFlow = hs_solver_jacobi.HornSchunckJacobi(params['alpha'], params['num_iterations'])

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 120)
if cap.isOpened() == False:
  print 'Unable to open camera'
  exit()

ret, im1 = cap.read()
if ret == False:
  print 'Unable to read camera'
  exit()

im1_flt = float32(cv2.cvtColor(im1, cv.CV_RGB2GRAY)) / 255.0
height, width = im1_flt.shape
out_vid = cv2.VideoWriter('flow.avi', cv2.cv.CV_FOURCC(*'MJPG'), 25, (width, height));

import time
ts = time.time()
fpscount = 0
while ret is not False:
  ret, im2 = cap.read()
  im2_flt = float32(cv2.cvtColor(im2, cv.CV_RGB2GRAY)) / 255.0

  u = zeros((height, width), dtype=float32)
  v = zeros((height, width), dtype=float32)
  u, v = OpticalFlow.runMultiLevel(im1_flt, im2_flt, u, v, params['num_pyramid'], params['num_full'])
  flowimg = flow_mod.run(double(u), double(v), 6.0)
  cv2.imshow('Flow', cv2.resize(flowimg, (640,480)))
  cv2.imshow('Video', cv2.resize(im1_flt, (640,480)))
  key = cv2.waitKey(1) % 256
  if key == 113:
    break
  im1_flt = im2_flt.copy()
  fpscount += 1
  if fpscount % 10 == 0:
    fps = fpscount / (time.time() - ts)
    print 'FPS: ' + str(fps)
    fpscount = 0
    ts = time.time()

cap.release()
out_vid.release()
