import numpy as np
import cv2
import sys
from copy import deepcopy
import collections

def read_video():
    if not sys.argv[1]:
        sys.exit("Please enter video name.")
    vid = cv2.VideoCapture(sys.argv[1])
    if not vid:
        sys.exit("File not found.")
    else:
        return vid


def get_background():
    pass


def get_foreground(frame, subtractor, fg_mask, learn_rate = 0.1):
    fg_mask = subtractor.apply(frame, fg_mask, learn_rate)
    return fg_mask, subtractor


def noise_filtering(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def edges_detection(image, lower_theshold=100, upper_threshold=200):
    return cv2.Canny(image, lower_theshold, upper_threshold, apertureSize=5)


def show_images(*args):
    cv2.imshow('frame', np.hstack(args))


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i == row1-1 and j == row2-1:
                return False

def isIn( lane ):
    pass

def get_lane():
    pass

def show_pixel_point(event, x, y ,flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print x,y

if __name__ == "__main__":
    count = [0]*2
    video = read_video()
    fgmask = None
    subtractor = cv2.BackgroundSubtractorMOG(history=500, nmixtures=10, backgroundRatio=0.5, noiseSigma=20)

    kernel = np.ones((10, 10), np.uint8)

    lane = {}
    lane1 = {"upLeft": (155, 182), "upRight": (225, 182),
             "lowLeft": (123, 326), "lowRight": (232, 326),
             "is_empty": True, "pts": []}
    lane2 = {"upLeft": (227, 182), "upRight": (302, 182),
             "lowLeft": (234, 326), "lowRight": (356, 326),
             "is_empty": True, "pts": []}
    points = []
    points.append(np.array([[lane1["upLeft"]],
                            [lane1["upRight"]],
                            [lane1["lowRight"]],
                            [lane1["lowLeft"]]], np.int32))
    points.append(np.array([[lane2["upLeft"]],
                            [lane2["upRight"]],
                            [lane2["lowRight"]],
                            [lane2["lowLeft"]]], np.int32))
    laneIM = []
    laneIM.append(np.zeros((480, 640), np.uint8))
    laneIM.append(np.zeros((480, 640), np.uint8))
    laneContour = [None]*2
    for i in range(0, 2):
        cv2.fillPoly(laneIM[i], [points[i]], 255)
        cv2.imshow(str(i), laneIM[i])
        laneContour[i], hrc = cv2.findContours(laneIM[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', show_pixel_point)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        res = frame

        cv2.polylines(frame, [points[0]], True, (0, 255, 0), 3)
        cv2.polylines(frame, [points[1]], True, (255, 0, 0), 3)
        filtered_frame = noise_filtering(image=frame)
        if fgmask is None:
            fgmask = subtractor.apply(filtered_frame, 0.1)
            fgmask, subtractor = get_foreground(frame=filtered_frame, subtractor=subtractor, fg_mask=fgmask)
        else:
            fgmask, subtractor = get_foreground(frame=filtered_frame, subtractor=subtractor, fg_mask=fgmask)

        fgmask = cv2.dilate(fgmask,kernel,iterations =3)
        fgmask = cv2.erode(fgmask,kernel,iterations =1)
        fgg = deepcopy(fgmask)
        contours, hierarchy = cv2.findContours(fgg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        isIn = [False, False]
        for obj in contours:
            moment = cv2.moments(obj)
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            cv2.circle(res, (cx, cy), 3, (0, 0, 255), 4)
            pX, pY, w, h = cv2.boundingRect(obj)
            dist = [cv2.pointPolygonTest(laneContour[0][0], (cx, cy), False),
                    cv2.pointPolygonTest(laneContour[1][0], (cx, cy), False)]
            for i in range(0, 2):
                if dist[i] == 1:
                    isIn[i] = True
                    cv2.rectangle(res,(pX, pY), (pX+w, pY+h),( 0, 255, 0 ), 2)
                    if i == 0:
                        if lane1["is_empty"] == True:
                            lane1["is_empty"] = False
                            lane1["pts"].append( (cx, cy))
                            count[0] += 1
                        else:
                            lane1["pts"].insert(0, (cx, cy))
                    else:
                        if lane2["is_empty"] == True:
                            lane2["is_empty"] = False
                            lane2["pts"].append( (cx, cy))
                            count[1] += 1
                        else:
                            lane2["pts"].insert(0, (cx, cy))
                else:
                    cv2.rectangle(res, (pX, pY), (pX+w, pY+h), (255, 255, 0), 2)

        if isIn[0]:
            #import ipdb; ipdb.set_trace()
            cv2.polylines(res,np.int32([lane1["pts"]]), False, (0,255,0) ,3)
        else:
            lane1["is_empty"] = True
            lane1["pts"] = []
        if isIn[1]:
            #import ipdb; ipdb.set_trace()
            cv2.polylines(res,np.int32([lane2["pts"]]), False, (0,255,0) ,3)
        else:
            lane2["is_empty"] = True
            lane2["pts"] = []

        #cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        cv2.putText(res,str(count[0]),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2)
        cv2.putText(res,str(count[1]),(450,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2)
        show_images(res)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.imwrite('testt.png',res)
            break
    print count
    video.release()
    cv2.destroyAllWindows()