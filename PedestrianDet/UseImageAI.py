from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

