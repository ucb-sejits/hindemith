import numpy as np
import cv2
NOISE_THRESHOLD = 30
NOISE_GAIN = 0.1

BLACK_THRESHOLD = NOISE_THRESHOLD * 1.25
MIN_HEIGHT = 60

offset = 0
FIXED_OFFSET = 1

VERTICAL_TOLERANCE = 30



def get_boxes(mag, image=None):
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
