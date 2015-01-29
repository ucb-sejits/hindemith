from optical_flow.hs_jacobi import HS_Jacobi
from box import get_boxes
from detect import run
import cv2
import numpy as np
from ctree.util import Timer
from time import sleep

hog = cv2.HOGDescriptor()

# hog = cv2.HOGDescriptor(
#    (48, 96), (16, 16), (8, 8), (8, 8), 9, 1, -1,
#    cv2.HOGDESCRIPTOR_L2HYS, 0.2, True, cv2.HOGDESCRIPTOR_DEFAULT_NLEVELS)

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rw < qw and rh < qh


def draw_detections(frame, windows):
    for i, win1 in enumerate(windows):
        y_start = max(win1[0] - 20, 0)
        y_end = max(win1[2] + 20, frame.shape[0])
        x_start = max(win1[1] - 10, 0)
        x_end = max(win1[3] + 10, frame.shape[1])
        window = frame[y_start:y_end, x_start:x_end]
        if window.shape < (128, 64):
            window = cv2.resize(window, (64, 128))

        found, w = hog.detect(
            window, winStride=(8, 8), padding=(32, 32))
        # found, w = hog.detectMultiScale(
        #     window, winStride=(8, 8), padding=(32, 32), scale=1.05)
        if len(found) > 0:
            cv2.rectangle(
                frame, (win1[1], win1[0]),
                (win1[3], win1[2]), (0, 255, 0), 2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", action='store_true', help="")
    parser.add_argument("--webcam", action='store_true', help="")
    args = parser.parse_args()
    if args.webcam:
        cap = cv2.VideoCapture(0)
        ret = cap.set(3, 384)
        ret = cap.set(4, 288)
        solver = HS_Jacobi(1, .5)
        prev_gray = None
        while(True):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                hsv = np.zeros_like(frame)
                continue
            else:
                u = solver(prev_gray, gray)
                mag, ang = cv2.cartToPolar(u[0], u[1])
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                ang = ang*180/np.pi/2
                hsv[..., 1] = 255
                hsv[..., 0] = ang
                hsv[..., 2] = mag
                flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # windows = []
                # for start in range(0, 120, 60):
                #     curr = np.copy(mag)
                #     curr[np.where(ang <= start)] = 0
                #     curr[np.where(ang >= start + 60)] = 0
                #     windows.extend(get_boxes(curr, flow))
                windows = get_boxes(mag, flow)
                # draw_detections(frame, windows)
                if len(windows) > 0:
                    run(np.array(windows), frame)
                frame = cv2.resize(frame, (640, 480))
                flow = cv2.resize(flow, (640, 480))
                cv2.imshow("frame", frame)
                cv2.imshow("flow", flow)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                prev_gray = None
    # elif args.sequence:
    else:
        import glob
        frame0 = None
        solve = HS_Jacobi(1, .5)
        for i, filename in enumerate(
                sorted(glob.glob('images/sequence1/image*.png'))):
            if frame0 is None:
                frame0 = cv2.imread(filename)
                hsv = np.zeros_like(frame0)
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                continue
            frame1 = cv2.imread(filename)

            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            with Timer() as t:
                u = solve(frame0, gray)
            print("flow time: {}".format(t.interval))
            mag, ang = cv2.cartToPolar(u[0], u[1])
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            ang = ang*180/np.pi/2
            hsv[..., 1] = 255
            hsv[..., 0] = ang
            hsv[..., 2] = mag
            flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            with Timer() as t:
                windows = get_boxes(mag, flow)
                # windows = []
                # for start in range(0, 120, 90):
                #     curr = np.copy(mag)
                #     curr[np.where(ang <= start)] = 0
                #     curr[np.where(ang >= start + 90)] = 0
                #     windows.extend(get_boxes(curr, flow))
            print("box time: {}".format(t.interval))

            with Timer() as t:
                # draw_detections(frame1, windows)
                run(np.array(windows), frame1)
            print("detect time: {}".format(t.interval))
            # cv2.imwrite('tmp/frame{0:05d}.jpg'.format(i), frame1)
            cv2.imwrite('tmp/flow{0:05d}.jpg'.format(i), flow)
            cv2.imshow("frame", frame1)
            cv2.imshow("flow", flow)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            frame0 = gray
    # else:
    #     frame0 = cv2.imread('images/frame0.png')
    #     frame1 = cv2.imread('images/frame1.png')
    #     im0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    #     im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #     solve = HS_Jacobi(1, .5)

    #     u = solve(im0, im1)

    #     mag, ang = cv2.cartToPolar(u[0], u[1])
    #     mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #     ang = ang*180/np.pi/2
    #     hsv = np.zeros_like(frame1)
    #     hsv[..., 1] = 255
    #     hsv[..., 0] = ang
    #     hsv[..., 2] = mag
    #     flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #     windows = get_boxes(mag, flow)
    #     # draw_detections(frame, windows)
    #     if len(windows) > 0:
    #         run(np.array(windows), frame1)
    #     cv2.imshow('detection', frame1)
    #     cv2.imshow('flow', flow)
    #     cv2.waitKey(0) & 0xff
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
