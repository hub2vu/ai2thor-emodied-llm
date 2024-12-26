from ai2thor.controller import Controller
import cv2 as cv
import time

controller = Controller(scene="FloorPlan1")
event = controller.step(action="RotateRight")
metadata = event.metadata
#print(event, event.metadata.keys())

#cv.waitKey(0)
for i in range(100):
  time.sleep(1)
  controller.step(action="RotateRight")
