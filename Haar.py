import cv2

# 使用人脸识别分类器
classfier = cv2.CascadeClassifier("C:/.xml")

# 读取图片
image = cv2.imread("face.jpg")
# 转为灰度图
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = classfier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(3,3))
print("发现{0}个人脸!".format(len(faces)))

for faceRect in faces:
    x,y,w,h=faceRect
    cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
cv2.imwrite("./face1.jpg",image)