import cv2
import csv
import sys

from random import randint

import numpy as np

fp = open('../report.csv', mode='w')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_TRIPLEX
TEXT_SIZE = 1.2
VIDEO_SOURCE = 'videos/people.mp4'
TITLE_TEXT_POSITION = (100, 40)

BGS_TYPES = ['GMG', 'MOG', 'MOG2', 'KNN', 'CNT']


def get_bgsubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8)
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True, varThreshold=100)
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                        maxPixelStability=15*60, isParallel=True)
    print('Invalid detector!')
    sys.exit(1)


cap = cv2.VideoCapture(VIDEO_SOURCE)

bg_subtractor = []
for i, a in enumerate(BGS_TYPES):
    bg_subtractor.append(get_bgsubtractor(a))


def main():
    framecount = 0
    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print('Finished processing the video')
            break

        framecount += 1
        frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)

        gmg = bg_subtractor[0].apply(frame)
        mog = bg_subtractor[1].apply(frame)
        mog2 = bg_subtractor[2].apply(frame)
        knn = bg_subtractor[3].apply(frame)
        cnt = bg_subtractor[4].apply(frame)

        gmg_count = np.count_nonzero(gmg)
        mog_count = np.count_nonzero(mog)
        mog2_count = np.count_nonzero(mog2)
        knn_count = np.count_nonzero(knn)
        cnt_count = np.count_nonzero(cnt)

        writer.writerow({'Frame': 'MOG', 'Pixel Count': mog_count})
        writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog_count})
        writer.writerow({'Frame': 'GMG', 'Pixel Count': mog_count})
        writer.writerow({'Frame': 'KNN', 'Pixel Count': mog_count})
        writer.writerow({'Frame': 'CNT', 'Pixel Count': mog_count})

        cv2.putText(mog, 'MOG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(mog2, 'MOG2', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(gmg, 'GMG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(knn, 'KNN', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(cnt, 'CNT', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('Original', frame)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        cv2.imshow('KNN', knn)
        cv2.imshow('CNT', cnt)
        cv2.imshow('GMG', gmg)

        cv2.moveWindow('Original', 0, 0)
        cv2.moveWindow('MOG', 0, 250)
        cv2.moveWindow('KNN', 0, 500)
        cv2.moveWindow('GMG', 719, 0)
        cv2.moveWindow('MOG2', 719, 250)
        cv2.moveWindow('CNT', 719, 500)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break


main()
