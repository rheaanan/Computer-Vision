import cv2
import numpy as np
import math

from eventBasedAnimationClass import EventBasedAnimationClass
from Tkinter import *
from GesturesApi import GestureProcessor
from PIL import Image, ImageTk
import argparse
# Import statements from: 
# http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter



# Subclasses Object from here:
# http://www.cs.cmu.edu/~112/notes/eventBasedAnimationClass.py
class GestureDemo(EventBasedAnimationClass):
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument('--video', default=False, action='store_true',help='use for video?')
        args = ap.parse_args()
        is_video = int(args.video)

        self.gp = GestureProcessor("Gesture_data.txt",is_video)  # default to usual file
        
        self.width = 1366
        self.height = 768
        super(GestureDemo, self).__init__(width=self.width, height=self.height)
        self.timerDelay = 1000 / 30 # 30 FPS
        # self.bindGestures()
        self.CVHandles = []
        self.bgHandle = None
        self.trackCenter = False
        self.trail = False
        

    def onTimerFired(self):
        self.gp.process()
        



    # OpenCV Image drawing adapted from:
    # http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter
    def drawCVImages(self):
        for handle in self.CVHandles:
            self.canvas.delete(handle)
        self.CVHandles = []

        cv2image = GestureProcessor.getRGBAFromBGR(self.gp.real,
                                                   self.width / 2,
                                                   self.height / 2)
        self.imagetk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        self.gp.draw()
        cv2image = GestureProcessor.getRGBAFromBGR(self.gp.drawingCanvas,
                                                   self.width / 2,
                                                   self.height / 2)
        self.imagetk2 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        cv2image = GestureProcessor.getRGBAFromGray(self.gp.thresholded,
                                                    self.width / 2,
                                                    self.height / 2)
        self.imagetk3 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        self.CVHandles.append(self.canvas.create_image(0, 0, image=self.imagetk,
                              anchor="nw"))
        self.CVHandles.append(self.canvas.create_image(1366, 768,
                              image=self.imagetk2, anchor="se"))
        self.CVHandles.append(self.canvas.create_image(0, 768,
                              image=self.imagetk3, anchor="sw"))

        self.CVHandles.append(self.canvas.create_text(1366, 0,
                              text=self.gp.lastAction, anchor="ne",
                              font="15"))
        self.CVHandles.append(self.canvas.create_text(1366, 15,
                              text="Distance: " + str(round(
                                                      self.gp.handDistance, 3)),
                              anchor="ne", font="15"))
        self.CVHandles.append(self.canvas.create_text(1366, 35,
                              text=str(self.gp.getScaledCenter()),
                              anchor="ne", font="15"))
    
    def drawBG(self):
        self.bgHandle = self.canvas.create_rectangle(self.width/2, 0,
                                                     self.width, self.height/2,
                                                     fill="white")

    def redrawAll(self):
        self.drawCVImages()

    def run(self):
        super(GestureDemo, self).run()
        self.onClose()

    def onClose(self):
        self.gp.close()  # MUST DO THIS

GestureDemo().run()

