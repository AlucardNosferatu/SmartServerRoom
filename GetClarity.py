import cv2
import os


def get_clarity():
    path = "Samples"
    for e, i in enumerate(os.listdir(path)):
        if i.endswith(".jpg") and (("(5)" in i) or ("(1)" in i)):
            img = cv2.imread(os.path.join(path, i), 1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            image_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()  # 图像模糊度
            # if int(image_var) < 200:
            print('>>>', i)
            print(">>>", int(image_var))


def blurify():
    path = "Samples"
    for e, i in enumerate(os.listdir(path)):
        for size in [5, 11, 21]:
            if i.endswith(".jpg"):
                img = cv2.imread(os.path.join(path, i), 1)
                blur = cv2.GaussianBlur(img, (size, size), 0)
                cv2.imwrite(os.path.join(path, i).replace(".jpg", "_blur" + str(size) + ".jpg"), blur)


get_clarity()
