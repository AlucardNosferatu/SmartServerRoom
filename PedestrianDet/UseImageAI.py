from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath('Models/resnet50_coco_best_v2.0.1.h5')
detector.loadModel()

