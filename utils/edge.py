import cv2
import numpy as np

def compute_edge_scharr(rgb):

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Scharr(gray, cv2.CV_64F,1,0)
    gy = cv2.Scharr(gray, cv2.CV_64F,0,1)

    return np.sqrt(gx**2 + gy**2)


def postprocess_edge(edge):

    edge = edge / (edge.max()+1e-8)
    bw = (edge >= 0.4).astype("uint8")

    kernel = np.ones((3,3),np.uint8)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)