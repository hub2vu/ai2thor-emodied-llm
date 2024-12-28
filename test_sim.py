from ai2thor.controller import Controller
import cv2 as cv
import time

controller = Controller(scene="FloorPlan1")
event = controller.step(action="RotateRight")
metadata = event.metadata
#print(event, event.metadata.keys())

#cv.waitKey(0)
# for i in range(1):
#   time.sleep(1)
#   event = controller.step(action="RotateRight")
#   print(event.metadata['agent'])
controller.step(action="Initialize", renderInstanceSegmentation=True)
for i in range(100):
  time.sleep(100)