from AVCS import AVCS
import sys
import cv2

if __name__ == "__main__":
    counter = AVCS()
    counter.readVideo(sys.argv[1])
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', counter.showPoint)
    counter.sampleImage()
    counter.setLane(0, (155, 182), (225, 182), (123, 326), (232, 326))
    counter.setLane(1, (227, 182), (302, 182), (234, 326), (356, 326))
    counter.run(cntStatus=True)
