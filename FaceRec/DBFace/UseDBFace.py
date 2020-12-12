import cv2
import common
from main import HAS_CUDA, detect
from model.DBFace import DBFace

DB_Face = DBFace()
DB_Face.eval()


def reload_weights():
    global DB_Face
    DB_Face.load("model/dbface.pth")


reload_weights()


def test_detector(frame):
    if HAS_CUDA:
        DB_Face.cuda()
    objs = detect(DB_Face, frame)
    rects = []
    for obj in objs:
        if obj.score > 0.5:
            rects.append([int(c) for c in obj.box])
    return rects


def test_landmarks(frame):
    if HAS_CUDA:
        DB_Face.cuda()
    objs = detect(DB_Face, frame)
    obj = objs[0]
    points_of_faces = [[int(c[0]), int(c[1])] for c in obj.landmark]
    return points_of_faces


def camera_demo():
    if HAS_CUDA:
        DB_Face.cuda()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()

    while ok:
        objs = detect(DB_Face, frame)
        rects = []
        for obj in objs:
            if obj.score > 0.5:
                rects.append(obj.box)
                common.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_demo()
