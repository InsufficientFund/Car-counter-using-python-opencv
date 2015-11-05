import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class AVCS:
    """docstring for ClassName"""
    def __init__(self):
        self.count = [0] * 2
        self.video = None
        self.fgMask = None
        self.subtractor = cv2.BackgroundSubtractorMOG(history=500, nmixtures=10, backgroundRatio=0.5, noiseSigma=20)
        self.lane = {"0": {"upLeft": (0, 0), "upRight": (0, 0),
                       "lowLeft": (0, 0), "lowRight": (0, 0),
                       "is_empty": True, "pts": []},
                      "1" : {"upLeft": (0, 0), "upRight": (0, 0),
                       "lowLeft": (0, 0), "lowRight": (0, 0),
                       "is_empty": True, "pts": []}}
        self.laneIm = [np.zeros((480, 640), np.uint8),
                        np.zeros((480, 640), np.uint8)]
        self.laneContour = [None] * 2
        self.points = [None] * 2
        self.sizeCar =[[], []]
        self.typeCar = {"small": 0, "medium": 0, "large": 0}

    def __del__(self):
        pass

    def readVideo(self, input):
        self.video = cv2.VideoCapture(input)

    def setLane(self, laneNum, upLeft, upRight, lowLeft, lowRight):
        self.lane[str(laneNum)] = {"upLeft": upLeft, "upRight": upRight,
                                   "lowLeft": lowLeft, "lowRight": lowRight,
                                   "is_empty": True, "pts": []}
        self.points[laneNum] = np.array([[self.lane[str(laneNum)]["upLeft"]],
                                        [self.lane[str(laneNum)]["upRight"]],
                                        [self.lane[str(laneNum)]["lowRight"]],
                                        [self.lane[str(laneNum)]["lowLeft"]]], np.int32)
        cv2.fillPoly( self.laneIm[laneNum], [self.points[laneNum]], 255)
        self.laneContour[laneNum], hrc = cv2.findContours( self.laneIm[laneNum], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    def showPoint(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print x, y

    def sampleImage(self):
        if self.video.isOpened():
            self.video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, 0)
            ret, frame = self.video.read()
        if ret:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def run(self, cntStatus = True, saveVid = False, showVid = True ):
        self.video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, 0)
        kernel = np.ones((10, 10), np.uint8)
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        testW = cv2.VideoWriter('/home/sayong/videos.avi',fourcc,15,(640, 480))
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            res = frame
            cv2.polylines(frame, [self.points[0]], True, (0, 255, 0), 3)
            cv2.polylines(frame, [self.points[1]], True, (125, 0, 255), 3)
            filteredFrame = cv2.GaussianBlur(frame, (5, 5), 0)
            if self.fgMask is None:
                self.fgMask = self.subtractor.apply(filteredFrame, 0.1)           
            self.fgMask = self.subtractor.apply(filteredFrame, self.fgMask, 0.1)
            self.fgMask = cv2.dilate(self.fgMask, kernel, iterations = 3)
            self.fgMask = cv2.erode(self.fgMask, kernel, iterations = 1)
            tempMask = deepcopy(self.fgMask)
            contours, hrc = cv2.findContours(tempMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            isIn = [False]*2
            for obj in contours:
                moment = cv2.moments(obj)
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                cv2.circle(res, (cx, cy), 3, (0, 0, 255), 4)
                pX, pY, w, h = cv2.boundingRect(obj)

                distance = [cv2.pointPolygonTest(self.laneContour[0][0], (cx, cy), False),
                            cv2.pointPolygonTest(self.laneContour[1][0], (cx, cy), False)]
                for i in range(0, 2):
                    if distance[i] == 1:
                        isIn[i] = True
                        cv2.rectangle(res, (pX, pY), (pX+w, pY+h), (0, 255, 255), 2)
                        if self.lane[str(i)]["is_empty"]:
                            self.lane[str(i)]["is_empty"] = False
                            self.lane[str(i)]["pts"].append((cx, cy))
                            self.count[i] += 1
                            self.sizeCar[i].append(w*h )
                            if w*h < 1600:
                                self.typeCar["small"] += 1
                            elif w*h < 9500:
                                self.typeCar["medium"] += 1
                            else:
                                self.typeCar["large"] += 1

                        else: 
                            self.lane[str(i)]["pts"].insert(0, (cx, cy))
                        break
                    else:
                        cv2.rectangle(res, (pX, pY), (pX+w, pY+h), (255, 255, 0), 2)
            for i in range(0, 2):
                if isIn[i]:
                    if showVid:
                        cv2.polylines(res, np.int32([self.lane[str(i)]["pts"]]), False, (0, 255, 255), 3)
                else:
                    self.lane[str(i)]["is_empty"] = True
                    self.lane[str(i)]["pts"] = []
            if cntStatus:
                cv2.putText(res, 'lane1: '+str(self.count[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(res, 'lane2: '+str(self.count[1]), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 255), 2)
                cv2.putText(res, 'truck/bus: '+str( self.typeCar["large"]), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(res, 'small car: '+str( self.typeCar["medium"]), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(res, 'motorcycle: '+str( self.typeCar["small"]), (400, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if showVid:
                cv2.imshow('frame', res)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            testW.write(res)
        print self.count
        self.video.release()
        testW.release()
        cv2.destroyAllWindows()
        print self.typeCar
        # plt.figure(1)
        # plt.plot(self.sizeCar[0],self.sizeCar[0], 'r*',self.sizeCar[1], self.sizeCar[1], 'rs')
        # plt.show()
        z = np.hstack((np.array(self.sizeCar[0]), np.array(self.sizeCar[1])))
        z = z.reshape((z.size,1))
        print 'test'
        z = np.float32(z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # Apply KMeans
        compactness,labels,centers = cv2.kmeans(z,3,criteria,10,flags)

        #plt.hist(z,256,[0,256]),plt.show()
        A = z[labels==0]
        B = z[labels==1]
        C = z[labels==2]
        plt.hist(A,256,None,edgecolor = 'r')
        plt.hist(B,256,None,edgecolor = 'b')
        plt.hist(C,256,None,edgecolor = 'g')
        plt.hist(centers,32,None,color = 'y')
        plt.show()
        import ipdb; ipdb.set_trace()
