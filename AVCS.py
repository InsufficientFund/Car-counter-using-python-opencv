import cv2
import numpy as np
from copy import deepcopy
from skimage import feature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 2),
            range=(0, self.numPoints + 1))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist, lbp

class AVCS:
    """docstring for ClassName"""
    def __init__(self):
        self.count = [0] * 2
        self.video = None
        self.fgMask = None
        self.sampleFrame = None
        #self.subtractor = cv2.BackgroundSubtractorMOG(history=150, nmixtures=20, backgroundRatio=0.7, noiseSigma=25)
        self.subtractor = cv2.BackgroundSubtractorMOG2(150, 200, False)
        self.lane = {"0": {"upLeft": (0, 0), "upRight": (0, 0),
                           "lowLeft": (0, 0), "lowRight": (0, 0),
                     "is_empty": True, "pts": []},
                     "1": {"upLeft": (0, 0), "upRight": (0, 0),
                           "lowLeft": (0, 0), "lowRight": (0, 0),
                           "is_empty": True, "pts": []}}
        self.laneIm = [np.zeros((480, 640), np.uint8),
                        np.zeros((480, 640), np.uint8)]
        self.laneContour = [None] * 2
        self.points = [None] * 2
        self.sizeCar =[[], []]
        self.typeCar = {"small": 0, "medium": 0, "large": 0}
        self.lbp = LocalBinaryPatterns(8, 1)

    def __del__(self):
        pass

    def readVideo(self, inputName):
        self.video = cv2.VideoCapture(inputName)

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
            ret, self.sampleFrame = self.video.read()
        if ret:
            cv2.imshow('frame', self.sampleFrame)
            cv2.waitKey(0)

    def writeClusters(self, data, members):
        dataF = open('data.csv', 'a')
        for cluster in range(3):
            for atr1, atr2 in data[members[cluster]]:
                dataF.write(str(atr1) + ", " + str(atr2) + ", " + str(cluster) + "\n")
        dataF.close()

    def getBackground(self, data, avg):
        cv2.accumulateWeighted(data, avg, 0.01)
        res = cv2.convertScaleAbs(avg)
        return res

    def run(self, cntStatus = True, saveVid = False, showVid = True ):
        self.video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, 0)
        kernel = np.ones((10, 10), np.uint8)
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        vidWriter = cv2.VideoWriter('/home/sayong/videos.avi', fourcc, 15, (640, 480))

        avg = np.float32(self.sampleFrame)
        lane1 = []
        lane2 = []
        total1 = 0
        total2 = 0
        dataPlot = []
        test = None
        while self.video.isOpened():
            currentObj = []
            ret, frame = self.video.read()
            if not ret:
                break
            bgFrame = self.getBackground(frame, avg)
            cv2.imshow('background', bgFrame)
            frameOrigin = deepcopy(frame)
            res = frame
            cv2.polylines(frame, [self.points[0]], True, (0, 255, 0), 3)
            cv2.polylines(frame, [self.points[1]], True, (125, 0, 255), 3)
            filteredFrame = cv2.GaussianBlur(frame, (5, 5), 0)
            if self.fgMask is None:
                self.fgMask = self.subtractor.apply(filteredFrame, -1)
                test = deepcopy(self.fgMask)
            self.fgMask = self.subtractor.apply(filteredFrame,self.fgMask, -1)
            #test = deepcopy(self.fgMask)
            self.fgMask = cv2.dilate(self.fgMask, kernel, iterations=1)
            self.fgMask = cv2.erode(self.fgMask, kernel, iterations=1)

            self.fgMask = cv2.morphologyEx(self.fgMask, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
            self.fgMask = cv2.morphologyEx(self.fgMask, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
            self.fgMask = cv2.morphologyEx(self.fgMask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            tempMask = deepcopy(self.fgMask)
            contours, hrc = cv2.findContours(tempMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            isIn = [False]*2
            laneObj = [[], []]
            outLane1 = []
            outLane2 = []
            for obj in contours:
                moment = cv2.moments(obj)
                if moment['m00'] == 0:
                    continue
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                pX, pY, w, h = cv2.boundingRect(obj)
                if cv2.pointPolygonTest(self.laneContour[0][0], (cx, cy), False) == 1:
                    carObj = {"centroid": (cx, cy+h/2), "origin": (pX, pY), "height": h, "width": w}
                    laneObj[0].append(carObj)
                elif cv2.pointPolygonTest(self.laneContour[1][0], (cx, cy), False) == 1:
                    carObj = {"centroid": (cx, cy+h/2), "origin": (pX, pY), "height": h, "width": w}
                    laneObj[1].append(carObj)
                elif cx >= 123 and cx <= 232 and cy >= 285 and cy <= 336 :
                    carObj = {"centroid": (cx, cy+h/2), "origin": (pX, pY), "height": h, "width": w}
                    outLane1.append(carObj)
                elif cx >= 232 and cx <= 356 and cy >= 285 and cy <= 336 :
                    carObj = {"centroid": (cx, cy+h/2), "origin": (pX, pY), "height": h, "width": w}
                    outLane2.append(carObj)

            for i in outLane1:
                diff1 = 51
                fObj1 = None
                for j in lane1:
                    diff = math.fabs(j["point"][0][0] - i["centroid"][0]) + math.fabs(j["point"][0][1] - i["centroid"][1])
                    if diff < diff1:
                        diff1 = diff
                        fObj1 = j
                if fObj1 != None:
                    total1 += 1
                    dataPlot.append(i["height"] * i["width"])
                    if i["height"] * i["width"] < 2500:
                        self.typeCar["small"] += 1
                    elif i["height"] * i["width"] < 25000:
                        self.typeCar["medium"] += 1
                    else:
                        self.typeCar["large"] += 1
                    originX = i["origin"][0]
                    originY = i["origin"][1]
                    crop_img = frameOrigin[originY:originY + i["height"], originX:originX+i["width"]]
                    normalImage = cv2.resize(crop_img, (64, 64))
                    grayImg = cv2.cvtColor(normalImage, cv2.COLOR_BGR2GRAY)
                    hist, lbp = self.lbp.describe(grayImg)
                    equ = cv2.equalizeHist(grayImg)
                    cv2.imwrite('/home/sayong/car/lane5'+str(total1)+'.png', equ)
                    cv2.imwrite('/home/sayong/car1/lane5'+str(total1)+'.png', crop_img)
                    lane1.remove(fObj1)

            for i in lane1:
                i["stat"] = False
            for i in laneObj[0]:
                diff1 = 51
                fObj1 = None
                for j in lane1:
                    #import ipdb;ipdb.set_trace()
                    diff = math.fabs(j["point"][0][0] - i["centroid"][0]) + math.fabs(j["point"][0][1] - i["centroid"][1])

                    if diff < diff1:
                        diff1 = diff
                        fObj1 = j
                if fObj1 != None:
                    fObj1["point"].insert(0, i["centroid"])
                    fObj1["stat"] = True
                else:
                    lane1.append({ "point": [i["centroid"]], "stat": True })
            tempLane =[]
            for i in lane1:
                if i["stat"]:
                    tempLane.append(i)
                    cv2.polylines(res, np.int32([i["point"]]), False, (0, 255, 255), 3)
            lane1 = tempLane

            for i in outLane2:
                diff2 = 51
                fObj2 = None
                for j in lane2:
                    diff = math.fabs(j["point"][0][0] - i["centroid"][0]) + math.fabs(j["point"][0][1] - i["centroid"][1])
                    if diff < diff2:
                        diff2 = diff
                        fObj2 = j
                if fObj2 != None:
                    total2 += 1
                    dataPlot.append(i["height"] * i["width"])
                    if i["height"] * i["width"] < 2500:
                        self.typeCar["small"] += 1
                    elif i["height"] * i["width"] < 25000:
                        self.typeCar["medium"] += 1
                    else:
                        self.typeCar["large"] += 1
                    originX = i["origin"][0]
                    originY = i["origin"][1]
                    crop_img = frameOrigin[originY:originY + i["height"], originX:originX+i["width"]]
                    normalImage = cv2.resize(crop_img, (64, 64))
                    grayImg = cv2.cvtColor(normalImage, cv2.COLOR_BGR2GRAY)
                    hist, lbp = self.lbp.describe(grayImg)
                    equ = cv2.equalizeHist(grayImg)
                    cv2.imwrite('/home/sayong/car/lane6'+str(total2)+'.png', equ)
                    cv2.imwrite('/home/sayong/car1/lane6'+str(total2)+'.png', crop_img)
                    lane2.remove(fObj2)
            for i in lane2:
                i["stat"] = False
            for i in laneObj[1]:
                diff2 = 51
                fObj2 = None
                for j in lane2:
                    #import ipdb;ipdb.set_trace()
                    diff = math.fabs(j["point"][0][0] - i["centroid"][0]) + math.fabs(j["point"][0][1] - i["centroid"][1])

                    if diff < diff2:
                        diff2 = diff
                        fObj2 = j
                if fObj2 != None:
                    fObj2["point"].insert(0, i["centroid"])
                    fObj2["stat"] = True
                else:
                    lane2.append({ "point": [i["centroid"]], "stat": True })
            tempLane =[]
            for i in lane2:
                if i["stat"]:
                    tempLane.append(i)
                    cv2.polylines(res, np.int32([i["point"]]), False, (0, 255, 255), 3)
            lane2 = tempLane


            for obj in contours:
                moment = cv2.moments(obj)
                if moment['m00'] == 0:
                    continue


                pX, pY, w, h = cv2.boundingRect(obj)
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])+h/2
                cv2.circle(res, (cx, cy), 3, (0, 0, 255), 4)
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
                            self.sizeCar[i].append([w/float(h), w*h/cv2.contourArea(self.laneContour[i][0])])
                            # if w*h < 1600:
                            #     self.typeCar["small"] += 1
                            # elif w*h < 9500:
                            #     self.typeCar["medium"] += 1
                            # else:
                            #     self.typeCar["large"] += 1

                        else:
                            self.lane[str(i)]["pts"].insert(0, (cx, cy))
                        break
                    else:
                        cv2.rectangle(res, (pX, pY), (pX+w, pY+h), (255, 255, 0), 2)
            for i in range(0, 2):
                if isIn[i]:
                    if showVid:
                        #cv2.polylines(res, np.int32([self.lane[str(i)]["pts"]]), False, (0, 255, 255), 3)
                        pass
                else:
                    self.lane[str(i)]["is_empty"] = True
                    self.lane[str(i)]["pts"] = []
            if cntStatus:
                #cv2.putText(res, 'lane1: '+str(self.count[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.putText(res, 'lane2: '+str(self.count[1]), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 255), 2)
                cv2.putText(res, 'lane1: '+str(total1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(res, 'lane2: '+str(total2), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 255), 2)
                cv2.putText(res, 'truck/bus: '+str( self.typeCar["large"]), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(res, 'small car: '+str( self.typeCar["medium"]), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(res, 'motorcycle: '+str( self.typeCar["small"]), (400, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if showVid:
                resMask = cv2.bitwise_and(frame, frame, mask=~self.fgMask)
                cv2.imshow('frame', res)
                vidWriter.write(res)
                if cv2.waitKey(8) & 0xFF == ord('q'):
                    cv2.imwrite('tesf.png', frameOrigin)
                    cv2.imwrite('tesM.png', self.fgMask)
                    break

        print self.count
        self.video.release()
        vidWriter.release()
        cv2.destroyAllWindows()
        print self.typeCar
        #print currentObj
        # totalAtr = np.array(self.sizeCar[0] + self.sizeCar[1])
        # k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
        # k_means.fit(totalAtr)
        # k_means_labels = k_means.labels_
        # k_means_cluster_centers = k_means.cluster_centers_
        # k_means_labels_unique = np.unique(k_means_labels)
        # colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        # plt.figure()
        #
        # plt.hold(True)
        # for k, col in zip(range(3), colors):
        #     members = k_means_labels == k
        #     cluster_center = k_means_cluster_centers[k]
        #     plt.plot(totalAtr[members, 0], totalAtr[members, 1], 'w',
        #              markerfacecolor=col, marker='.')
        #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=6)
        # plt.title('KMeans')
        # plt.grid(True)
        # plt.show()
        plt.figure(3)
        plt.plot(dataPlot)
        plt.ylabel('some numbers')
        plt.show()
        # print dataPlot
        # members = []
        # for cluster in range(3):
        #     member = k_means_labels == cluster
        #     members.append(member)
        # self.writeClusters(totalAtr, members)