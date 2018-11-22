import cv2
import numpy as np
import math

from eventBasedAnimationClass import EventBasedAnimationClass
from Tkinter import *
from GesturesApi import GestureProcessor
from PIL import Image, ImageTk
# Import statements from: 
# http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter



# Subclasses Object from here:
# http://www.cs.cmu.edu/~112/notes/eventBasedAnimationClass.py
class GestureDemo(EventBasedAnimationClass):
    def __init__(self):
        self.gp = GestureProcessor("Gesture_data.txt")  # default to usual file
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


'''
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # read image
    ret, img = cap.read()

    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    crop_img = img[100:300, 100:300]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # show thresholded image
    cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(crop_img,start, end, [0,255,0], 2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)

    # define actions required
    if count_defects == 1:
        cv2.putText(img,"Play", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(img,"Fwd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img,"Rewind", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"No result", (50, 50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break
'''