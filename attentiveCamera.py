import sys, time
import numpy as np
import cv2
from cv2 import cv
from util import RectsSet, Counter
from operator import sub
from math import hypot

FACE_CASCADE = r".\cascades\haarcascade_frontalface_alt.xml"
HAND_CASCADE = r".\cascades\1256617233-1-haarcascade_hand.xml"
TIME_DELTA = 0.5
UPDATE_TIME = 10
THRESHHOLD = 0.01
FACES = 0
HANDS = 1
gFrameCount = -1
gCounter = Counter()
gFaceDict = dict()



def detect(img, cascade, flag) :
    """detect faces/hands from image by using the Viola-Jones algorithm."""
    min_size = (0,0)
    if flag == FACES :
        rects = cascade.detectMultiScale(img, 1.2, 3, minSize = min_size)
    elif flag == HANDS :
        rects = cascade.detectMultiScale(img, 1.2, 5)
    if len(rects) == 0 :
        return RectsSet()
    rects[:,2:] += rects[:,:2]
    
    return RectsSet([tuple(rect) + (time.clock(), ) for rect in rects])

def drawRects(img, rects, color) :
    """draw bounding rectangules around detected objects."""
    for x1, y1, x2, y2, t in rects:
        cv2.rectangle(img, (x1,y1) , (x2,y2), color, 2)

def checkFaceHandRelation(coveredFaces, faces, hands, histFaces) :
    """check if hand location is approximetely close to a previously detected face location."""
    global gCounter, gFaceDict
    remove = set([])
    faceRemove = set([])
    handRemove = set([])
    histFacesRemove = set([])
    tmpTime = time.clock()
    #go over all faces which were previously covered
    for face in coveredFaces :
        covered = False
        #check if current face is still covered by newly detected hands
        for hand in hands:
            faceCenter = ( int( (face[0] + face[2])/ 2.0 ), int ((face[1] + face[3])/ 2.0 ) )
            if (hand[0] < faceCenter[0] < hand[2] and hand[1] < faceCenter[1] < hand[3] and hand[4] > face[4]):
                    covered = True
        #if face is not covered, start countdown for revieling face.
        if not covered:
            gCounter[face] += 1
            if gCounter[face] == 2 :
                remove.add(face)
                histFaces.add(face)
                gCounter.pop(face)
        else:
            gCounter[face] = 0
    coveredFaces.difference_update(remove)

    #go over all newly detected hands and faces, update faces which are covered
    for hand in hands:
        for face in faces: 
            faceCenter = ( int( (face[0] + face[2])/ 2.0 ), int ((face[1] + face[3])/ 2.0 ) )
            if (hand[0] < faceCenter[0] < hand[2] and hand[1] < faceCenter[1] < hand[3]):
                    coveredFaces.add(face)
            
        for face in histFaces :
            faceCenter = ( int( (face[0] + face[2])/ 2.0 ), int ((face[1] + face[3])/ 2.0 ) )
            if (hand[0] < faceCenter[0] < hand[2] and hand[1] < faceCenter[1] < hand[3] and hand[4] > face[4]):
                coveredFaces.add(face)
                histFacesRemove.add(face)

        if tmpTime - hand[4] > TIME_DELTA :
            handRemove.add(hand)
    
    
    for face in histFaces :
        if tmpTime - face[4] > 8 * TIME_DELTA :
            histFacesRemove.add(face)

    for face in faces :
        #TIME_DELTA is multiplied in order to allow for faces just recently covered to stay.
        if tmpTime - face[4] > 4*TIME_DELTA :
                faceRemove.add(face)
    faces.difference_update(faceRemove)
    hands.difference_update(handRemove)
    histFaces.difference_update(histFacesRemove)
            
def run() :
    """run of main program"""
    global gFrameCount, gFaceDict
    
    current = time.clock()
    #use Viola-Jones cascades from training set to detect faces/hands
    faceCascade = cv2.CascadeClassifier(FACE_CASCADE)
    handCascade = cv2.CascadeClassifier(HAND_CASCADE)
    
    capture = cv2.VideoCapture(0)
    
    retVal, frame = capture.read()
    height, width =  frame.shape[:2]
    faces = set([])
    hands = set([])
    coveredFaces = set([])
    histFaces = set([])
    while True: 
        retVal, frame = capture.read()
        cv2.imshow('background', cv2.flip(frame, 1) )
        k = cv2.waitKey(3)
        if k == 0x20 :
            break
    bg = frame
    tmpGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('frame', flags = cv2.WINDOW_NORMAL)
    while True:
        retVal, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmpTime = time.clock()
        #program updates background image (for cropping out person) every UPDATE_TIME minutes,
        #given that there is not much movement in frame with respect to last "snapshot".
        if tmpTime - current > UPDATE_TIME :
            #check if the difference between current frame to frame captured UPDATE_TIME
            #minutes ago is small - and if so: update background image.
            err = cv2.norm(gray, tmpGray, cv2.NORM_L2)/gray.size
            if err < THRESHHOLD :
                bg = frame
                cv2.imshow('background', cv2.flip(bg, 1) )
            tmpGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tmpTime = current = time.clock()
           
        gray = cv2.equalizeHist(gray)

        #run the Viola-Jones algorithm to detect new faces/hands in frame
        faces.update(detect(gray, faceCascade, FACES) )
        hands.update(detect(gray, handCascade, HANDS) )
        checkFaceHandRelation(coveredFaces, faces, hands, histFaces)
        
        #for debugging:
        drawRects(frame, faces, (128,128,128) )
        drawRects(frame, hands, (128,128,255) )   

        #crop out all covered faces
        output = frame[:]
        for face in coveredFaces:
            w = face[2] - face[0]
            h = face[3] - face[1]
            x1 = max(0, int(face[0]-1.5*w))
            x2 = min(frame.shape[1], int(face[2] + 1.5*w) )
            y1 = max(0, int(face[1]-h))
            y2 = min(frame.shape[0] - 2, int(face[3]+9*h) )
            output[y1:y2, x1:x2] = bg[y1:y2, x1:x2]
            
        output = cv2.flip(output,1)
        cv2.imshow('frame', output)

        k = cv2.waitKey(1)
        if k == 0x1b: # ESC
            print 'ESC pressed. Exiting ...'
            break
    capture.release()
 
    cv2.destroyAllWindows()   
    sys.exit(1)

if __name__ == "__main__" :
    print "Press ESC to exit"
    run()
