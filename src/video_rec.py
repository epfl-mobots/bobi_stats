import rospy
import cv2
from subprocess import Popen, DEVNULL, PIPE


class VideoRec:
    def __init__(self, filename, width, height, fps, codec='libx264', pixfmt='yuv420p'):
        self._filename = filename
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._pixfmt = pixfmt

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self._vr = cv2.VideoWriter(
            self._filename + '.mp4', fourcc, fps, (width, height))

    def write(self, img, t):
        cv2.putText(img,
                    'Log time: {:.3f}'.format(t),
                    (int(img.shape[1]*0.55), int(img.shape[0]*0.93)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 200),
                    2,
                    cv2.LINE_AA)
        self._vr.write(img)

    def __del__(self):
        self._vr.release()
