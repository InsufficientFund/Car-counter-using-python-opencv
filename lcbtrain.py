import glob
import cv2
import numpy as np
from copy import deepcopy
from skimage import feature
import csv
import re

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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]




lbpClass = LocalBinaryPatterns(8, 1)
listFile = glob.glob("/home/sayong/Project/Source/carData/car/*.png")
listDetialFile = glob.glob("/home/sayong/Project/Source/carData/car1/*.png")
n = 0
listRange = [slice(0, 11), slice(11, 22), slice(22, 33),
             slice(33, 44), slice(44, 54), slice(54, 64)]
featureList = []
# histRatio =[]
# histMax =[]
sizeList = []

listFile.sort(key=natural_keys)
listDetialFile.sort(key=natural_keys)

for fileP in listDetialFile:
    rgbPic = cv2.imread(fileP)
    height, width, channels = rgbPic.shape
    sizeData = [height/100.0, width/100.0, height * width/10000.0]
    sizeList.append(sizeData)

index = 0
for fileP in listFile:

    print fileP
    pic = cv2.imread(fileP)
    grayPic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    histRatio = []
    histMax = []
    for i in listRange:
        for j in listRange:
            n += 1
            computePic = grayPic[i, j]
            hist, lbp = lbpClass.describe(computePic)
            histR = hist[0:5].sum()/hist[5:10].sum()
            histRatio.append(round(histR, 6))
            histMax.append(round(hist.max(),6))
            #import ipdb; ipdb.set_trace()
    featureList.append(np.hstack((histRatio, histMax, sizeList[index])))
    index += 1

with open('validated_raw.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(featureList)
fp.close()
#print featureList