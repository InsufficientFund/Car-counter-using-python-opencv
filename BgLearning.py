import numpy as np
import cv2
import sys


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


def get_foreground(frame, subtractor, fg_mask, learn_rate = 0.05):
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

if __name__ == "__main__":
    video = read_video()
    fgmask = None
    while video.isOpened():
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if fgmask is None:
            fgbg = cv2.BackgroundSubtractorMOG()
            fgmask = fgbg.apply(frame)
            fgmask,fgbg = get_foreground(frame=gray, subtractor=fgbg, fg_mask=fgmask)
        else:
            fgmask,fgbg = get_foreground(frame=gray, subtractor=fgbg, fg_mask=fgmask)

        fgmask = noise_filtering(image=fgmask)
        edges = edges_detection(image=fgmask, lower_theshold=100, upper_threshold=200)

        dilation = cv2.dilate(edges, (5,5), iterations=1)


        #-----------------------------------------------------------------------------------------------
        contours,hier = cv2.findContours(edges, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

        # import ipdb;ipdb.set_trace()
        if hier is not None:
            true_len = len(contours)
            LENGTH = len(cnts)
            status = np.zeros((LENGTH, 1))
            if true_len <= 100:
                print true_len
                for i, cnt1 in enumerate(cnts):
                    x = i
                    if i != LENGTH-1:
                        for j, cnt2 in enumerate(cnts[i+1:]):
                            x = x+1
                            dist = find_if_close(cnt1,cnt2)
                            if dist is True:
                                val = min(status[i],status[x])
                                status[x] = status[i] = val
                            else:
                                if status[x] == status[i]:
                                    status[x] = i+1

                unified = []
                maximum = int(status.max())+1
                for i in xrange(maximum):
                    pos = np.where(status == i)[0]
                    if pos.size != 0:
                        cont = np.vstack(cnts[i] for i in pos)
                        hull = cv2.convexHull(cont)
                        unified.append(hull)

                # cv2.drawContours(gray,unified,-1,(0,255,0),2)

                cv2.drawContours(fgmask,unified,-1,255,-1)
        # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #-----------------------------------------------------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        res = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
        # cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        show_images(fgmask, gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

