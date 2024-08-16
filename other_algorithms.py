import cv2
import sys

import numpy as np
from random import randint


TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))

print(TEXT_COLOR)
print(BORDER_COLOR)

FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/traffic_4.mp4'
BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CMT']
print(BGS_TYPES)


def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel


get_kernel('opening')


def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, get_kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation


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
    sys.exit(0)


cap = cv2.VideoCapture(VIDEO_SOURCE)
# 0 = GMG, 1 = MOG2, 2 = MOG, 3 = KNN, 4 = CNT
bg_subtractor = get_bgsubtractor(BGS_TYPES[2])
BGS_TYPE = BGS_TYPES[2]


def main():
    while cap.isOpened():
        ok, frame = cap.read()
        # print(ok)
        # print(frame.shape)

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        bg_mask = bg_subtractor.apply(frame)
        fg_mask = get_filter(bg_mask, 'dilation')
        fg_mask_closing = get_filter(bg_mask, 'closing')
        fg_mask_opening = get_filter(bg_mask, 'opening')
        fg_mask_combine = get_filter(bg_mask, 'combine')

        res = cv2.bitwise_and(frame, frame, mask=fg_mask)
        res_closing = cv2.bitwise_and(frame, frame, mask=fg_mask_closing)
        res_opening = cv2.bitwise_and(frame, frame, mask=fg_mask_opening)
        res_combine = cv2.bitwise_and(frame, frame, mask=fg_mask_combine)

        cv2.putText(res_combine, 'Background subtractor: ' + BGS_TYPE, (10, 50),
                    FONT, 1, BORDER_COLOR, 3, cv2.LINE_AA)

        cv2.putText(res_combine, 'Background subtractor: ' + BGS_TYPE, (10, 50),
                    FONT, 1, BORDER_COLOR, 3, cv2.LINE_AA)
        if not ok:
            print('End processing the video')
            break

        if BGS_TYPE != 'MOG' and BGS_TYPE != 'GMG':
            cv2.imshow('Background model', bg_subtractor.getBackgroundImage())

        cv2.imshow('Frame', frame)
        # cv2.imshow('BG Mask', bg_mask)
        # cv2.imshow('Dilation', fg_mask)
        cv2.imshow('Dilation final', res)
        cv2.imshow('Closing final', res_closing)
        cv2.imshow('Opening final', res_opening)
        cv2.imshow('Combine final', res_combine)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()
