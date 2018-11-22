import cv2

def nothing(x): #needed for createTrackbar to work in python.
    pass    

cap = cv2.VideoCapture(0)
cv2.namedWindow('temp')
cv2.createTrackbar('bl', 'temp', 0, 255, nothing)
cv2.createTrackbar('gl', 'temp', 0, 255, nothing)
cv2.createTrackbar('rl', 'temp', 0, 255, nothing)
cv2.createTrackbar('bh', 'temp', 255, 255, nothing)
cv2.createTrackbar('gh', 'temp', 255, 255, nothing)
cv2.createTrackbar('rh', 'temp', 255, 255, nothing)
while True:
        ret,img=cap.read()#Read from source
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        bl_temp=cv2.getTrackbarPos('bl', 'temp')
        gl_temp=cv2.getTrackbarPos('gl', 'temp')
        rl_temp=cv2.getTrackbarPos('rl', 'temp')
        bh_temp=cv2.getTrackbarPos('bh', 'temp')
        gh_temp=cv2.getTrackbarPos('gh', 'temp')
        rh_temp=cv2.getTrackbarPos('rh', 'temp')
        thresh=cv2.inRange(hsv,(bl_temp,gl_temp,rl_temp),(bh_temp,gh_temp,rh_temp))
        if(cv2.waitKey(10) & 0xFF == ord('b')):
            break #break when b is pressed 
        cv2.imshow('Video', img)
        cv2.imshow('thresh', thresh)