__author__ = 'chick'

import numpy as np


class CACGUpdate(object):
    def kernel(self, coef, P, R, ):
        out_shape = P.shape[:2]
        s = P.shape[2]
        x = np.zeros(out_shape)
        p = np.zeros(out_shape)
        r = np.zeros(out_shape)

        for in_x in range(P.shape[0]):
            for in_y in range(P.shape[1]):
                x_f = p_f = r_f = 0.0

                for i in range(s):
                    p_f += coef[i, 0] * P[in_x, in_y, i]
                    r_f += coef[i, 1] * P[in_x, in_y, i]
                    x_f += coef[i, 2] * P[in_x, in_y, i]

                for i in range(s):
                    v = i + s
                    p_f += coef[v, 0] * R[in_x, in_y, i]
                    r_f += coef[v, 1] * R[in_x, in_y, i]
                    x_f += coef[v, 2] * R[in_x, in_y, i]

                x[in_x, in_y] += x_f
                p[in_x, in_y] += p_f
                r[in_x, in_y] += r_f

        return x, p, r

if __name__ == '__main__':
    s = 5
    shape = [16, 8, s]
    shape2 = [16, 8]
    P = np.random.random(shape)
    R = np.random.random(shape)
    coef = np.random.random([2*s, 3])

    x, p, r = CACGUpdate().kernel(coef, P, R)

    print x



    


"""
  for(int in_x = block_id0 * blksize + get_local_id(0) + bdr ; in_x < (block_id0+1) * blksize - bdr ; in_x += get_local_size(0)) {
    for(int in_y = block_id1 * blksize + get_local_id(1) + bdr ; in_y < (block_id1+1) * blksize - bdr ; in_y += get_local_size(1)) {
      int out_x = in_x - (block_id0 * 2*bdr) - bdr;
      int out_y = in_y - (block_id1 * 2*bdr) - bdr;
      float x_f = 0.0f;
      float p_f = 0.0f;
      float r_f = 0.0f;
      for(int i = 0 ; i < s+1 ; i++) {
          p_f += coef->get("i", "0")*P->get("in_x", "in_y", "i");
          r_f += coef->get("i", "1")*P->get("in_x", "in_y", "i");
          x_f += coef->get("i", "2")*P->get("in_x", "in_y", "i");
      }
      for(int i = 0 ; i < s ; i++) {
          vs = i+s+1;
          p_f += coef->get(vs.str(), "0")*R->get("in_x", "in_y", "i");
          r_f += coef->get(vs.str(), "1")*R->get("in_x", "in_y", "i");
          x_f += coef->get(vs.str(), "2")*R->get("in_x", "in_y", "i");
      }
      x->get("out_x", "out_y") += x_f;
      p->get("out_x", "out_y") = p_f;
      r->get("out_x", "out_y") = r_f;
    }
  }
"""