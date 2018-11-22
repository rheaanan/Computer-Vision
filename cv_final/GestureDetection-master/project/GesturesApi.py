import cv2
import numpy as np
import time
import os
import defaultGesturesLoader
from gesture import Gesture
import random
import math
import pyautogui


class GestureProcessor(object):
    def __init__(self, gestureFile="gestureData.txt"):
        self.cap = cv2.VideoCapture(0)
        self.cameraWidth = 1366
        self.cameraHeight = 768
        #self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
        #self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
        self.stationary = False
        self.record = False
        self.endGesture = False
        self.gesturePoints = []
        #self.gestureFile = gestureFile
        self.gestureHeader = "Gesture Name: "
        self.gestureEnd = "END GESTURE"
        #self.saveNextGesture = False
        self.lastAction = ""
        self.handMomentPositions = []
        self.handCenterPositions = []
        self.initGestures()

# --------------------------------- Gesture IO --------------------------------
# Functions associated with loading and saving the gestures, either from the
# default gestures or the file provided.

    def initGestures(self):
        self.loadDefaultGestures()

    # Initiate some default gesures in the event that no gesture file was found
    def loadDefaultGestures(self):
        self.gestures = defaultGesturesLoader.defaultGestures
        self.gestureNames = []
        for gesture in self.gestures:
            self.gestureNames.append(gesture.name)

    def getGestureNames(self):
        return [gesture.name for gesture in self.gestures]


# ------------------------------ Image Processing ------------------------------
# Functions associated with reading the image from the camera, modifying it
# to make processing easier, and ultimately extracting the contour.

    def readCamera(self):
        _, self.real = self.cap.read()
        self.real = cv2.flip(self.real, 1)
        cv2.rectangle(self.real, (300,300), (100,100), (0,255,0),0)
        self.original = self.real[100:300, 100:300]

        

    def threshold(self):
        grey = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        value = (31, 31)
        blurred = cv2.GaussianBlur(grey, value, 0)
        #self.thresholded=cv2.inRange(blurred,(255,255,255),(29,255,132));
        _, self.thresholded = cv2.threshold(blurred, 127, 255,
                                            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
       

    def extractContours(self):
        self.contours, _ = cv2.findContours(self.thresholded.copy(),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    # Currently just finds the largest contour,
    # Should be able to replace this with a "matching" algorithm from here:
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/
    #py_contours_more_functions/py_contours_more_functions.html
    def extractHandContour(self):

        self.cnt = max(self.contours, key = lambda x: cv2.contourArea(x))
        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(self.cnt)
        cv2.rectangle(self.original, (x, y), (x+w, y+h), (0, 0, 255), 0)
        maxArea, index = 0, 0
        for i in xrange(len(self.contours)):
            area = cv2.contourArea(self.contours[i])
            if area > maxArea:
                maxArea = area
                index = i
        self.realHandContour = self.contours[index]
        self.realHandLen = cv2.arcLength(self.realHandContour, True)
        # reduce hand contour to manageable number of points
        # Thanks to http://opencvpython.blogspot.com/2012/06/
        #                                           contours-2-brotherhood.html
        self.handContour = cv2.approxPolyDP(self.realHandContour,
                                            0.001 * self.realHandLen, True)

# ----------------------------- Contour Processing -----------------------------
# Functions to process the contour to determine various data, such as
# center, width, height, distance, etc.

    def setHandDimensions(self):
        self.minX, self.minY, self.handWidth, self.handHeight = \
            cv2.boundingRect(self.cnt)

    def findHullAndDefects(self):
        self.hullHandContour = cv2.convexHull(self.cnt)
        #self.hullPoints = [self.cnt[i[0]] for i in self.hullHandContour]
        #self.hullPoints = np.array(self.hullPoints, dtype = np.int32)
        self.hullHand = cv2.convexHull(self.cnt, returnPoints=False)
        self.defects = cv2.convexityDefects(self.cnt,
                                            self.hullHand)

    # Documentation:
    # http://docs.opencv.org/doc/tutorials/imgproc/shapedescriptors/moments/
    # moments.html
    def findCenterWithMoments(self):
        self.handMoments = cv2.moments(self.handContour)
        self.handXCenterMoment = int(self.handMoments["m10"] /
                                     self.handMoments["m00"])
        self.handYCenterMoment = int(self.handMoments["m01"] /
                                     self.handMoments["m00"])
        self.handMoment = (self.handXCenterMoment, self.handYCenterMoment)
        self.handMomentPositions += [self.handMoment]

    # Credit for this algorithm goes to the paper which can be found at the
    # description in this link: https://www.youtube.com/watch?v=xML2S6bvMwI
    def centerWithReduction(self):
        scaleFactor = 0.3
        shrunk = np.array(self.handContour * scaleFactor, dtype=np.int32)
        tx, ty, w, h = cv2.boundingRect(shrunk)
        maxPoint = None
        maxRadius = 0
        for x in xrange(w):
            for y in xrange(h):
                rad = cv2.pointPolygonTest(shrunk, (tx + x, ty + y), True)
                if rad > maxRadius:
                    maxPoint = (tx + x, ty + y)
                    maxRadius = rad
        realCenter = np.array(np.array(maxPoint) / scaleFactor,
                                  dtype=np.int32)
        error = int((1 / scaleFactor) * 1.5)
        maxPoint = None
        maxRadius = 0
        for x in xrange(realCenter[0] - error, realCenter[0] + error):
            for y in xrange(realCenter[1] - error, realCenter[1] + error):
                rad = cv2.pointPolygonTest(self.handContour, (x, y), True)
                if rad > maxRadius:
                    maxPoint = (x, y)
                    maxRadius = rad
        return np.array(maxPoint)

    def findCenterCircleAndRadius(self):
        self.palmCenter = self.centerWithReduction()
        self.palmRadius = cv2.pointPolygonTest(self.cnt,
                                               tuple(self.palmCenter), True)
        self.handCenterPositions += [tuple(self.palmCenter)]

    def getDistance(self):
        self.handDistance = (self.cameraWidth + self.cameraHeight) / \
            float(self.palmRadius)

    def analyzeHandCenter(self):
        # makes sure that there is actually sufficient data to trace over
        if len(self.handCenterPositions) > 10:
            self.recentPositions = sorted(self.handCenterPositions[-30:])
            self.x = [pos[0] for pos in self.recentPositions]
            self.y = [pos[1] for pos in self.recentPositions]
        else:
            self.recentPositions = []

# ----------------------------- Gesture Detection -----------------------------
# Functions associated with determining gestures

    def checkCanDoGestures(self):
        if len(self.handCenterPositions) > 10:
            self.canDoGestures = True
        else:
            self.canDoGestures = False

    def detemineStationary(self):
        # Figure out of the past few points have been at roughly same position
        # If they have and there is suddenly movement,
        # trigger the start of a gesture search
        searchLength = 3 # 3 frames should be enough
        val = -1 * (searchLength + 1)
        self.prevRecordState = self.record
        if self.canDoGestures:
            xPoints = [pt[0] for pt in self.handMomentPositions[val:-1]]
            yPoints = [pt[1] for pt in self.handMomentPositions[val:-1]]
            xAvg = np.average(xPoints)
            yAvg = np.average(yPoints)
            factor = 0.04
            for x, y in self.handMomentPositions[-(searchLength + 1):-1]:
                # if any point is further further from the average:
                if (x-xAvg)**2 + (y-yAvg)**2 > factor * min(self.cameraWidth,
                                                            self.cameraHeight):
                    # If previous not moving, start recording
                    if self.stationary:
                        self.record = True
                    self.stationary = False
                    self.stationaryTimeStart = time.time()
                    return
            # Not previously stationary but stationary now
            if not self.stationary:
                self.record = False
            self.stationary = True

    def classifyGesture(self):
        minError = 2**31 - 1 # a large value
        minErrorIndex = -1
        self.humanGesture = Gesture(self.gesturePoints, "Human Gesture")
        likelihoodScores = [0] * len(self.gestures)
        assessments = [{}] * len(self.gestures)
        for i in xrange(len(self.gestures)):
            assessments[i] = Gesture.compareGestures(self.gestures[i],
                                                        self.humanGesture)
        errorList = [assessments[i][Gesture.totalError] \
                        for i in xrange(len(assessments))]
        index = errorList.index(min(errorList))
        # Basic elimination to figure out if result is valid


        # show appropriate images in windows

        templateGestureRatio = max((self.gestures[index].distance /\
                                    self.humanGesture.distance), 
                                    (self.humanGesture.distance /\
                                        self.gestures[index].distance))
        distanceDiffRatio = assessments[index][Gesture.totalDistance] /\
                                min(self.gestures[index].distance,
                                    self.humanGesture.distance)
        if templateGestureRatio < 1.25 and distanceDiffRatio < 2:
            return index
            

    def determineIfGesture(self):
        if self.stationary:
            self.timedcounter+=1
            if self.count_defects == 1:
                cv2.putText(self.real,"Play", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
               # pyautogui.press('space')
            elif self.count_defects == 2:
                cv2.putText(self.real,"Fwd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                #pyautogui.keyDown('alt');
                #pyautogui.press('right');
            elif self.count_defects == 3:
                cv2.putText(self.real,"Rewind", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                #pyautogui.keyDown('alt');
                #pyautogui.press('left');
                #pyautogui.keyUp('alt')
            elif self.count_defects == 4:
                cv2.putText(self.real,"Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            else:
                cv2.putText(self.real,"No result", (50, 50),\
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        if self.record:
            self.gesturePoints += [self.handCenterPositions[-1]]
        elif self.prevRecordState == True and not self.record:
            minGesturePoints = 5  # Should last a few frames at least
            if len(self.gesturePoints) > minGesturePoints:
                gestureIndex = self.classifyGesture()
                if gestureIndex != None:
                    self.gestures[gestureIndex].action()
                    self.lastAction = self.gestures[gestureIndex].name
                elif gestureIndex == None and self.saveNextGesture:
                    self.addRecordedGesture()
                    self.saveNextGesture = False
            self.gesturePoints = []

# --------------------------------- Main Loop ---------------------------------
# All of the processing is initiated from this function. Everything is laid
# out in the proper order and named so that the algorithm is easy to follow.

    # importantly, changed so that it works on a tick instead
    def process(self):
        self.readCamera()
        self.threshold()
        self.extractContours()
        self.extractHandContour()
        self.setHandDimensions()
        self.findHullAndDefects()
        self.findCenterWithMoments()
        self.findCenterCircleAndRadius()
        self.getDistance()
        self.analyzeHandCenter()
        self.checkCanDoGestures()
        self.detemineStationary()
        self.determineIfGesture()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

# --------------------------- Gestures API Functions --------------------------
# Various other things necessary to make this a more complete API.

    def getScaledCenter(self):
        return (self.palmCenter / np.array([self.cameraWidth,
                                            self.cameraHeight],
                                            dtype=np.float32)).round(3)

# ---------------------------------- Graphics ----------------------------------
# Functions associated with being able to draw the data in a human friendly
# way. Necessary to be able to see data in an external program.
    
    @staticmethod
    def getRGBAFromBGR(image, width, height):
        resized = cv2.resize(image, (width, height))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)

    @staticmethod
    def getRGBAFromGray(image, width, height):
        resized = cv2.resize(image, (width, height))
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGBA)

    def getRGBAThresh(self, widthScale=1, heightScale=1):
        if widthScale != 1 or heightScale != 1:
            resized = cv2.resize(self.thresholded, (0, 0), fx=widthScale,
                                    fy=heightScale)
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGBA)
        return cv2.cvtColor(self.thresholded, cv2.COLOR_GRAY2RGBA)

    def getRGBAOriginal(self, widthScale=1, heightScale=1):
        if widthScale != 1 or heightScale != 1:
            resized = cv2.resize(self.original, (480, 320))
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(self.original, cv2.COLOR_BGR2RGBA)

    def getRGBACanvas(self, widthScale=1, heightScale=1):
        if widthScale != 1 or heightScale != 1:
            resized = cv2.resize(self.drawingCanvas, (0, 0), fx=widthScale,
                                    fy=heightScale)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(self.drawingCanvas, cv2.COLOR_BGR2RGBA)

    def drawCenter(self):
        cv2.circle(self.drawingCanvas, tuple(self.palmCenter),
                   10, (255, 0, 0), -2)
        if len(self.recentPositions) != 0:
            for i in xrange(len(self.recentPositions)):
                cv2.circle(self.drawingCanvas, self.recentPositions[i], 5,
                           (255, 25*i, 25*i), -1)

    def drawCircles(self):
        cv2.circle(self.drawingCanvas, tuple(self.palmCenter),
                   int(self.palmRadius), (0, 255, 0), 10)

    def drawHandContour(self, bubbles = False):
        cv2.drawContours(self.drawingCanvas, [self.cnt],
                         0, (0, 255, 0), 1)

    def drawHullContour(self, bubbles = False):
        cv2.drawContours(self.drawingCanvas, [self.hullHandContour], 0,
                         (0, 0, 255), 0)


    def drawDefects(self, bubbles = False):
            self.count_defects = 0
            cv2.drawContours(self.thresholded, self.contours, -1, (0, 255, 0), 3)

		    # applying Cosine Rule to find angle for all defects (between fingers)
		    # with angle > 90 degrees and ignore defects
            for i in range(self.defects.shape[0]):
                s,e,f,d = self.defects[i,0]
                start = tuple(self.cnt[s][0])
                end = tuple(self.cnt[e][0])
                far = tuple(self.cnt[f][0])

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                # ignore angles > 90 and highlight rest with red dots
                if angle <= 90:
                    self.count_defects += 1
                    cv2.circle(self.real, far, 1, [0,0,255], 0)
                #dist = cv2.pointPolygonTest(cnt,far,True)
                # draw a line from start to end i.e. the convex points (finger tips)
                # (can skip this part)
                cv2.line(self.real,start, end, [0,255,0], 0)
                #cv2.circle(crop_img,far,5,[0,0,255],-1)

    def drawBubbles(self, pointsList, color=(255, 255, 255), width=2):
        for i in xrange(len(pointsList)):
            for j in xrange(len(pointsList[i])):
                cv2.circle(self.drawingCanvas, (pointsList[i][j][0],
                           pointsList[i][j][1]), width, color)

    def draw(self):
        self.drawingCanvas = np.zeros(self.original.shape, np.uint8)
        self.drawHandContour(True)
        self.drawHullContour(True)
        self.drawDefects(True)
        self.drawCenter()
        self.drawCircles()
