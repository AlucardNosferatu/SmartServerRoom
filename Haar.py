import cv2


def detect_3_parts(classifier_full, classifier_lower, classifier_upper, image2):
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    full_body = classifier_full.detectMultiScale(gray, scaleFactor=1.005, minNeighbors=5, minSize=(3, 3))
    lower_body = classifier_lower.detectMultiScale(gray, scaleFactor=1.005, minNeighbors=5, minSize=(3, 3))
    upper_body = classifier_upper.detectMultiScale(gray, scaleFactor=1.005, minNeighbors=5, minSize=(3, 3))

    print("发现{0}个全身!".format(len(full_body)))
    print("发现{0}个下身!".format(len(lower_body)))
    print("发现{0}个上身!".format(len(upper_body)))

    xc_fb = []
    xc_lb = []
    xc_ub = []
    for rect in full_body:
        x, y, w, h = rect
        xc_fb.append(x + int(w / 2))
        image2 = cv2.rectangle(image2, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for rect in lower_body:
        x, y, w, h = rect
        xc_lb.append(x + int(w / 2))
        image2 = cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for rect in upper_body:
        x, y, w, h = rect
        xc_ub.append(x + (w / 2))
        image2 = cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image2, full_body, lower_body, upper_body, xc_fb, xc_lb, xc_ub


def calculate_white_list(xc_fb, xc_lb, xc_ub, full_body, lower_body, upper_body, image):
    white_list = []
    for i, xc in enumerate(xc_fb):
        distance_lb = [abs(xc_l - xc) for xc_l in xc_lb]
        mdl = min(distance_lb)
        if mdl < 30:
            distance_ub = [abs(xc_u - xc) for xc_u in xc_ub]
            mdu = min(distance_ub)
            if mdu < 30:
                white_list.append([i, distance_lb.index(mdl), distance_ub.index(mdu)])
            else:
                pass
        else:
            pass
    for indices in white_list:
        c1 = transform_coordinates(full_body[indices[0]])
        c2 = transform_coordinates(lower_body[indices[1]])
        c3 = transform_coordinates(upper_body[indices[2]])
        c4 = merge_coordinates(c1, c2, c3)
        x, y, w, h = transform_coordinates(c4, inv=True)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return white_list, image


def transform_coordinates(c, inv=False):
    if inv:
        x1, y1, x2, y2 = c
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return [x, y, w, h]
    else:
        x, y, w, h = c
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return [x1, y1, x2, y2]


def merge_coordinates(c1, c2, c3):
    x1_list = [c1[0], c2[0], c3[0]]
    x1 = min(x1_list)
    y1_list = [c1[1], c2[1], c3[1]]
    y1 = min(y1_list)
    x2_list = [c1[2], c2[2], c3[2]]
    x2 = max(x2_list)
    y2_list = [c1[3], c2[3], c3[3]]
    y2 = max(y2_list)
    c4 = [x1, y1, x2, y2]
    return c4


if __name__ == "__main__":
    # 使用人脸识别分类器
    classifier_full = cv2.CascadeClassifier("Models/haarcascade_fullbody.xml")
    classifier_lower = cv2.CascadeClassifier("Models/haarcascade_lowerbody.xml")
    classifier_upper = cv2.CascadeClassifier("Models/haarcascade_upperbody.xml")
    # 读取图片
    image = cv2.imread("Samples/P (2).jpg")
    image2 = image.copy()
    # 转为灰度图
    detected = detect_3_parts(classifier_full, classifier_lower, classifier_upper, image2)
    image2, full_body, lower_body, upper_body, xc_fb, xc_lb, xc_ub = detected
    cv2.imshow('fuck2', image2)
    white_list, image = calculate_white_list(xc_fb, xc_lb, xc_ub, full_body, lower_body, upper_body, image)
    cv2.imshow('fuck', image)
    cv2.waitKey()
