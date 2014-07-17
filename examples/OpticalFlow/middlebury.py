import struct
import os
import numpy

def read_flo(fname):
  flo = open(fname, 'rb')
  flo.seek(4)
  width = struct.unpack("i", flo.read(4))[0]
  height = struct.unpack("i", flo.read(4))[0]
  u = numpy.zeros((height, width))
  v = numpy.zeros((height, width))
  for yy in range(height):
    for xx in range(width):
      u[yy,xx] = struct.unpack("f", flo.read(4))[0]
      v[yy,xx] = struct.unpack("f", flo.read(4))[0]
  return (u, v, width, height)

def aae(u, uref, v, vref, width, height):
  mask = (numpy.fabs(uref) < 1e9) * (numpy.fabs(vref) < 1e9)
  vangle = mask*(numpy.arctan2(v, u) - numpy.arctan2(vref, uref))
  mask = vangle > numpy.pi
  vangle = vangle - mask * 2 * numpy.pi
  mask = vangle < -numpy.pi
  vangle = vangle + mask * 2 * numpy.pi
  verror = numpy.sum(numpy.fabs(vangle))
  return verror / (width*height) * (180 / numpy.pi)
 
def calculate_aae(fname1, fname2):
  (u1, v1, width1, height1) = read_flo(fname1)
  (u2, v2, width2, height2) = read_flo(fname2)
  res = aae(u1, u2, v1, v2, width1, height1)
  return res

def epe(u, uref, v, vref, width, height):
  mask = (numpy.fabs(uref) < 1e9) * (numpy.fabs(vref) < 1e9)
  udiff = mask * (u - uref)
  vdiff = mask * (v - vref)
  verror = numpy.sqrt(udiff*udiff + vdiff*vdiff)
  return numpy.mean(verror)
 
def calculate_epe(fname1, fname2):
  (u1, v1, width1, height1) = read_flo(fname1)
  (u2, v2, width2, height2) = read_flo(fname2)
  res = epe(u1, u2, v1, v2, width1, height1)
  return res

def write_middlebury(x, fname, width, height):
  f = open('out.flo', 'wb')
  f.write("\x50\x49\x45\x48")
  width_bin = struct.pack("i", width)
  height_bin = struct.pack("i", height)
  f.write(width_bin)
  f.write(height_bin)
  for yy in range(height):
    for xx in range(width):
      u_bin = struct.pack("f", x[xx+yy*width])
      v_bin = struct.pack("f", x[xx+yy*width+width*height])
      f.write(u_bin)
      f.write(v_bin)
  f.close()
  #call_str = 'flow-code/color_flow out.flo ' + fname
  #os.system(call_str)

def write_middlebury_interleaved(flow_x, flow_y, width, height):
  fx = flow_x.flatten();
  fy = flow_y.flatten();
  x = numpy.dstack((fx,fy)).flatten();
  f = open('out.flo', 'wb')
  f.write("\x50\x49\x45\x48")
  width_bin = struct.pack("i", width)
  height_bin = struct.pack("i", height)
  f.write(width_bin)
  f.write(height_bin)
  ub = struct.pack('%sf' % len(x), *x);
  f.write(ub);
  f.close()
