import cv2

from EllipsesDetection import get_ellipse

if __name__ == "__main__":
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while sample.isOpened():
        ret, img = sample.read()
        if img is not None:
            get_ellipse(img)
