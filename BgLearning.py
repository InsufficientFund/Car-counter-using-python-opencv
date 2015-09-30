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

def is_pass_upper_line( lane ):
    pass



if __name__ == "__main__":
    video = read_video()
    fgmask = None
    subtractor = cv2.BackgroundSubtractorMOG(history=500, nmixtures=10, backgroundRatio=0.5, noiseSigma=20)
    kernel = np.ones((10,10),np.uint8)
    lane = {}
    lane1 = { "up1":(155,182) , "up2":(225, 182) ,"low1":(105, 394), "low2":(235, 394),"is_empty":True, "pts":[] }
    points = np.array( [ [lane1["up1"]], [lane1["up2"]] , [lane1["low2"]] , [lane1["low1"]] ],np.int32 )
    tempIM = np.zeros((480,640),np.uint8)
    cv2.fillPoly(tempIM,[points], 255)
    cv2.imshow("1", tempIM)
    laneContour, hrc = cv2.findContours(tempIM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #import ipdb; ipdb.set_trace()
    while video.isOpened():
        ret, frame = video.read()
        res = frame

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.line(gray, (215, 142), (389, 142), (255, 0, 0), 5)
        cv2.line(frame, (155, 182), (310, 182), (255, 0, 0), 5)
        cv2.line(frame, (105, 394), (410, 394), (255, 0, 0), 5)
        cv2.line(frame, (155, 182), (105, 394), (255, 0, 0), 5)
        cv2.line(frame, (225, 182), (235, 394), (255, 0, 0), 5)
        filtered_frame = noise_filtering(image=frame)
        if fgmask is None:
            fgmask = subtractor.apply(filtered_frame, 0.1)
            fgmask, subtractor = get_foreground(frame=filtered_frame, subtractor=subtractor, fg_mask=fgmask)
        else:
            fgmask, subtractor = get_foreground(frame=filtered_frame, subtractor=subtractor, fg_mask=fgmask)


        # edges = edges_detection(image=fgmask, lower_theshold=100, upper_threshold=200)
        #
        # dilation = cv2.dilate(edges, (5,5), iterations=1)
        #
        #
        # #-----------------------------------------------------------------------------------------------
        # contours,hier = cv2.findContours(edges, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        #
        # # import ipdb;ipdb.set_trace()
        # if hier is not None:
        #     true_len = len(contours)
        #     LENGTH = len(cnts)
        #     status = np.zeros((LENGTH, 1))
        #     if true_len <= 100:
        #         print true_len
        #         for i, cnt1 in enumerate(cnts):
        #             x = i
        #             if i != LENGTH-1:
        #                 for j, cnt2 in enumerate(cnts[i+1:]):
        #                     x = x+1
        #                     dist = find_if_close(cnt1,cnt2)
        #                     if dist is True:
        #                         val = min(status[i],status[x])
        #                         status[x] = status[i] = val
        #                     else:
        #                         if status[x] == status[i]:
        #                             status[x] = i+1
        #
        #         unified = []
        #         maximum = int(status.max())+1
        #         for i in xrange(maximum):
        #             pos = np.where(status == i)[0]
        #             if pos.size != 0:
        #                 cont = np.vstack(cnts[i] for i in pos)
        #                 hull = cv2.convexHull(cont)
        #                 unified.append(hull)
        #
        #         # cv2.drawContours(gray,unified,-1,(0,255,0),2)
        #
        #         cv2.drawContours(fgmask,unified,-1,255,-1)
        # # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # #-----------------------------------------------------------------------------------------------
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #res = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
        # cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask,kernel,iterations =3)
        fgmask = cv2.erode(fgmask,kernel,iterations =1)
        #import ipdb; ipdb.set_trace()
        fgg = deepcopy(fgmask)
        contours, hierarchy = cv2.findContours(fgg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        isIn = False
        for obj in contours:
            moment = cv2.moments(obj)
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            cv2.circle(res, (cx, cy), 3, (0, 0, 255), 4)
            pX, pY, w, h =  cv2.boundingRect(obj)
            dist = cv2.pointPolygonTest(laneContour[0],(cx, cy) ,False)
            if dist == 1:
                isIn = True
                cv2.rectangle(res,(pX, pY), (pX+w, pY+h),( 0, 255, 0 ), 2)
                if lane1["is_empty"] == True:
                    lane1["pts"].append( (cx, cy))
                else:
                    lane1["pts"].insert(0, (cx,cy))
            else:
                cv2.rectangle(res,(pX, pY), (pX+w, pY+h),( 255, 255, 0 ), 2)

        if isIn:
            #import ipdb; ipdb.set_trace()
            cv2.polylines(res,np.int32([lane1["pts"]]), False, (0,255,0) ,3)
        else:
            lane1["is_empty"] = True
            lane1["pts"] = []

        #cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        show_images(res)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()