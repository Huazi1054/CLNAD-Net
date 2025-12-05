import cv2
import os
import numpy as np
from Inpainter import Inpainter

path = r'./Artifact_label32/'
mask_path = r'./Artifact_mask32/'
# path = r'./Artifact_label/'
imgs = os.listdir(path)
good_path = r'./pre_images32_p/'

for i in range(0, len(imgs)):
    img = cv2.imread(path + imgs[i], cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # cv2.imshow('ori', img)
    # cv2.imshow('step1', Roberts)

    # Set threshold to filter out tissue artifacts
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if Roberts[m, n] < 100:
                Roberts[m, n] = 0
            else:
                Roberts[m, n] = 255

    # cv2.imshow('step2', Roberts)

    # Calculate connected domains
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Roberts, connectivity=8)

    # Calculate the size of the connected domain
    temp = [0 for _ in range(labels.max() + 1)]  # 0 is blackground
    for num in range(labels.max()):
        for m in range(img.shape[0]):
            for n in range(img.shape[1]):
                temp[labels[m, n]] += 1

    b = sorted(enumerate(temp), key=lambda temp: temp[1])
    c = [temp[0] for temp in b]
    print('length: %d' % (len(c)))
    print(c)
    if len(c) >= 35:
        c = c[-35:-1]
    else:
        c = c[:-1]

    if len(c) > 1:

        output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for m in range(img.shape[0]):
            for n in range(img.shape[1]):
                if labels[m, n] in c:
                    output[m, n] = 255

        # cv2.imshow('step4', output)

        # The mask begins to expand
        k1 = np.ones((3, 3), np.uint8)
        ans = cv2.dilate(output, k1, iterations=1)
        # cv2.imshow('step5', ans)
        # k2 = np.ones((3, 3), np.uint8)
        # ans = cv2.erode(ans, k2, iterations=1)
        # cv2.imshow('step6', ans)

        mask = 255 - ans
        marker = np.zeros_like(ans)
        marker[0, :] = 255
        marker[-1, :] = 255
        marker[:, 0] = 255
        marker[:, -1] = 255
        SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
        count = 0
        while True:
            count += 1
            marker_pre = marker
            # Expansion marker
            dilation = cv2.dilate(marker, kernel=SE)
            marker = np.min((dilation, mask), axis=0)

            # Determine whether the result after expansion is consistent with the previous iteration, and if so, complete the hole filling.
            if (marker_pre == marker).all():
                break

        marker = 255 - marker
        # cv2.imshow('step7', marker)
        cv2.imwrite(mask_path + imgs[i], marker)

        # 修复图像
        halfPatchWidth = 4
        iteration = Inpainter(img, marker, halfPatchWidth)
        if iteration.checkValidInputs() == iteration.CHECK_VALID:
            iteration.inpaint()
            # cv2.imshow('final', iteration.result)
            cv2.imwrite(good_path + imgs[i], iteration.result)

    print("Finish %d " % (i+1))
    # cv2.waitKey(0)
    # print(labels.max())