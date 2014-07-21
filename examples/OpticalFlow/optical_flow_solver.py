from numpy import zeros

import logging
from hindemith.core import *
from hindemith.operations.optical_flow.pyr_down import pyr_down, pyr_down_fn
from hindemith.operations.optical_flow.pyr_up import pyr_up

logging.basicConfig(level=20)

defines = {}

defines['nd20'] = 1
defines['nd21'] = 1
defines['nt20'] = 16
defines['nt21'] = 16
defines['ne20'] = 1
defines['ne21'] = 1

# defines['nd20'] = 1
#defines['nd21'] = 1
#defines['nt20'] = 32
#defines['nt21'] = 1
#defines['ne20'] = 1
#defines['ne21'] = 1


@fuse
def update_uv(u, v, du, dv, two, w, h):
    u = u + du
    v = v + dv
    u = u * two
    v = v * two
    #u = median_filter(u)
    #v = median_filter(v)
    new_u = pyr_up(u)
    new_v = pyr_up(v)
    return new_u, new_v


@fuse
def update_uv_noresize(u, v, du, dv):
    new_u = u + du
    new_v = v + dv
    #new_u = median_filter(new_u)
    #new_v = median_filter(new_v)
    return new_u, new_v


class OpticalFlowSolver(object):
    def runMultiLevel(self, im1, im2, u, v, num_pyr, num_full):
        self.bytes_transferred = 0
        self.theoretical_bytes_transferred = 0
        self.flops = 0
        # pyr1 = [Array2D.fromArray(im1)]
        # pyr2 = [Array2D.fromArray(im2)]
        # sizes = [(ScalarConstant(im1.shape[0]), ScalarConstant(im1.shape[1]))]
        pyr1 = [im1]
        pyr2 = [im2]
        sizes = [(im1.shape[0], im1.shape[1])]

        for i in range(1, num_pyr + 1):
            pyr1.insert(0, pyr_down_fn(im=pyr1[0]))
            pyr2.insert(0, pyr_down_fn(im=pyr2[0]))
            sizes.insert(0, (pyr1[0].shape[0], pyr1[0].shape[1]))

        hm_u = zeros(sizes[0], dtype=u.dtype)
        hm_v = zeros(sizes[0], dtype=v.dtype)
        two = 2.0

        # For each size
        for i in range(0, num_pyr):
            hm_du, hm_dv = self.run(pyr1[i], pyr2[i], hm_u, hm_v)
            hm_u, hm_v = update_uv(u=hm_u, v=hm_v, du=hm_du, dv=hm_dv, two=two, w=sizes[i + 1][0],
                                   h=sizes[i + 1][1])
            #sync(hm_u)
            #sync(hm_v)
            #import flow_mod
            #flowimg = flow_mod.run(float64(hm_u), float64(hm_v), 0.0)
            #cv2.imshow('flowimg', flowimg)
            #cv2.waitKey(10)

        # For each size
        for i in range(0, num_full):
            hm_du, hm_dv = self.run(pyr1[num_pyr], pyr2[num_pyr], hm_u, hm_v)
            hm_u, hm_v = update_uv_noresize(u=hm_u, v=hm_v, du=hm_du, dv=hm_dv)
            #sync(hm_u)
            #sync(hm_v)
            #import flow_mod
            #flowimg = flow_mod.run(float64(hm_u), float64(hm_v), 0.0)
            #cv2.imshow('flowimg', flowimg)
            #cv2.waitKey(10)

        # sync(hm_u)
        # sync(hm_v)
        # u = array(hm_u)
        # v = array(hm_v)
        u = hm_u.data
        v = hm_v.data
        return u, v
