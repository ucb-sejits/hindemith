import numpy as np
import os
file_path = os.path.dirname(os.path.realpath(__file__))


NOISE_THRESHOLD = 30
NOISE_GAIN = 0.1

BLACK_THRESHOLD = NOISE_THRESHOLD * 1.25
MIN_HEIGHT = 60

offset = 0
FIXED_OFFSET = 1

VERTICAL_TOLERANCE = 30

import cv2


def get_boxes(mag, image=None):

    # width = mag.shape[1]
    # height = mag.shape[0]
    # horiz_hist = []

    # for y in range(height):
    #     horiz_hist.append(np.max(mag[y]))

    # rows = []
    # start = None
    # for i, val in enumerate(horiz_hist):
    #     if val > NOISE_THRESHOLD:
    #         if start is None:
    #             # if len(rows) > 0 and i - rows[-1][1] < 5:
    #             #     start = rows.pop()[0]
    #             # else:
    #             start = i
    #     else:
    #         if start is not None:
    #             rows.append((start, i))
    #             start = None

    # ret = []
    # for row in rows:
    #     vert_hist = []
    #     for col in range(width):
    #         vert_hist.append(np.max(mag[row[0]:row[1], col]))

    #     start = None
    #     border = 3
    #     for i, val in enumerate(vert_hist):
    #         if val > NOISE_THRESHOLD:
    #             if start is None:
    #                 start = i
    #             border = 3
    #         elif border == 0:
    #             if start is not None:
    #                 row_start = row[0]
    #                 row_end = row[1]
    #                 for r in range(row[0], row[1]):
    #                     if np.max(mag[r, start:i]) < 30:
    #                         row_start += 1
    #                     else:
    #                         break
    #                 for r in reversed(range(row[0], row[1])):
    #                     if np.max(mag[r, start:i]) < 30:
    #                         row_end -= 1
    #                     else:
    #                         break
    #                 if row_end - row_start > 32 and i - start > 16:
    #                     ret.append([row_start, start, row_end, i])
    #                 start = None
    #         else:
    #             border -= 1


    width = mag.shape[1]
    height = mag.shape[0]
    horiz_hist = []

    for y in range(height):
        horiz_hist.append(np.max(mag[y]))

    rows = []
    start = None
    for i, val in enumerate(horiz_hist):
        if val > NOISE_THRESHOLD:
            if start is None:
                # if len(rows) > 0 and i - rows[-1][1] < 5:
                #     start = rows.pop()[0]
                # else:
                start = i
                thresh = 5
        else:
            if start is not None:
                thresh -= 1
                if thresh == 0:
                    rows.append((start, i))
                    start = None
    if start is not None:
        rows.append((start, height))

    ret = []
    # print(horiz_hist)
    # print(rows)
    for row in rows:
        # ret.append((row[0], 0, row[1], 384))
        vert_hist = []
        for col in range(width):
            vert_hist.append(np.max(mag[row[0]:row[1], col]))

        start = None
        for i, val in enumerate(vert_hist):
            if val > NOISE_THRESHOLD:
                if start is None:
                    start = i
            else:
                if start is not None:
                    row_start = row[0]
                    row_end = row[1]
                    if row_end - row_start > 16 and i - start > 16:
                        for r in range(row[0], row[1]):
                            if np.max(mag[r, start:i]) < 30:
                                row_start += 1
                            else:
                                break
                        for r in reversed(range(row[0], row[1])):
                            if np.max(mag[r, start:i]) < 30:
                                row_end -= 1
                            else:
                                break
                        row_start = max(row_start - 5, 0)
                        row_end = min(row_end + 5, height)
                        start = max(start - 5, 0)
                        i = min(i + 5, width)
                        ret.append([row_start, start, row_end, i])
                    start = None
        if start is not None:
            row_start = row[0]
            row_end = row[1]
            if row_end - row_start > 16 and i - start > 16:
                for r in range(row[0], row[1]):
                    if np.max(mag[r, start:i]) < 30:
                        row_start += 1
                    else:
                        break
                for r in reversed(range(row[0], row[1])):
                    if np.max(mag[r, start:i]) < 30:
                        row_end -= 1
                    else:
                        break
                row_start = max(row_start - 5, 0)
                row_end = min(row_end + 5, height)
                start = max(start - 5, 0)
                i = min(i + 5, width)
                ret.append([row_start, start, row_end, i])

    if image is not None:
        for rect in ret:
            cv2.rectangle(image, (rect[1], rect[0]),
                        (rect[3], rect[2]), (0, 0, 255), 2)

    return ret


"""
[('_temp/det_input.png', array([[150,  47, 409, 171],
       [146, 177, 392, 198],
       [149, 207, 398, 227],
       [159, 230, 409, 250],
       [164, 323, 244, 343],
       [154, 355, 305, 375],
       [165, 429, 244, 451],
       [166, 496, 315, 516],
       [154, 616, 252, 636],
       [150,  68, 374,  89],
       [146, 142, 410, 291],
       [154, 339, 305, 382],
       [165, 418, 250, 440],
       [163, 469, 329, 546],
       [159, 550, 244, 570]]))]
"""
