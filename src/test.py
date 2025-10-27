import cv2 as cv
import numpy as np
import open3d

from utils import *
import log_utils as log

# Loading the config file globally
with open("./config/config.yaml") as f:
    config = yaml.safe_load(f)


def PreprocessImages(imgList: list[np.array]) -> list[np.array]:
    
    clahe = cv.createCLAHE(clipLimit=config["Preprocessing"]["CLAHE"]["clipLimit"]) # Initializing a CLAHE object
    kernel_size = config["Preprocessing"]["GaussianBlur"]["KernelSize"]
    
    processedImgList = []
    for img in imgList:
        processedImg = cv.GaussianBlur(img, (kernel_size, kernel_size), 0) # Blurring the image
        processedImg = clahe.apply(processedImg) # Applying clahe
        processedImgList.append(processedImg)

    return processedImgList


def TestFeatureExtraction(container: BundleAdjustmentContainer, displayFlag: bool=True) -> None:
    
    # Getting the keypoints and descriptors
    container.kpList, container.desList = container.extractFeatures(container.grayList)
    
    # Drawing the keypoints
    drawnImgList = []
    for img, kp in zip(container.grayList, container.kpList):
        drawnImg = cv.drawKeypoints(img, kp, None)
        drawnImgList.append(drawnImg)

    # Displaying the images
    if displayFlag:
        scaleFactor = config["ScaleFactor"]
        DisplayImages(drawnImgList, (scaleFactor, scaleFactor))

    return None


def TestFeatureMatching(container: BundleAdjustmentContainer, displayFlag: bool=True) -> None:

    drawnImgList = []

    for idx in range(len(container.imgList) - 1):

        # Images
        img1 = container.grayList[idx]
        img2 = container.grayList[idx+1]

        # Descriptors
        kp1 = container.kpList[idx]
        kp2 = container.kpList[idx+1]
        
        # Finding matches
        matches = container.findMatches(idx, idx+1)

        # Drawn image
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        drawnImgList.append(img3)
    
    # Displaying the images
    if displayFlag:
        scaleFactor = config["ScaleFactor"]
        DisplayImages(drawnImgList, (scaleFactor, scaleFactor))
    
    return None

def TestComputeEssentialMatrixCompute(container: BundleAdjustmentContainer, displayFlag: bool=True) -> None:

    CAM_MATRIX = np.array([
                            [1662,    0, 540],
                            [   0, 1673, 960],
                            [   0,    0,   1]], dtype=np.float32)
    
    drawnImgList = []

    for idx in range(len(container.imgList) - 1):

        # Images
        img1 = container.grayList[idx]
        img2 = container.grayList[idx+1]

        # Descriptors
        kp1 = container.kpList[idx]
        kp2 = container.kpList[idx+1]
        
        # Finding matches
        matches = container.findMatches(idx, idx+1)

        if len(matches)<10:
            print("Not enough matches between image {} and image {}!".format(idx, idx+1))
            exit()
        
        # Extracting source pts and destination pts
        srcPts = np.float32([kp2[m.trainIdx].pt for [m] in matches]).reshape(-1,1,2)
        dstPts = np.float32([kp1[m.queryIdx].pt for [m] in matches]).reshape(-1,1,2)

        E, mask = cv.findEssentialMat(srcPts, dstPts, CAM_MATRIX, method=cv.RANSAC, threshold=3)

        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesMask=mask)
        drawnImgList.append(img3)

        log.info(f"Num matches between images {idx} and {idx+1}: {int(sum(mask))}")
    
        # Displaying the images
    if displayFlag:
        scaleFactor = config["ScaleFactor"]
        DisplayImages(drawnImgList, (scaleFactor, scaleFactor))
    
    return None





if __name__ == "__main__":
    
    # Initializing the container
    container = BundleAdjustmentContainer(config)
    
    # Loading the images
    container.imgList, container.grayList = LoadImages("data/")

    # Preprocessing the images
    container.grayList = PreprocessImages(container.grayList)

    # Testing keypoints
    TestFeatureExtraction(container, False)

    # Testing matches
    TestFeatureMatching(container, False)

    # Testing Essential Matrix computation and filtering matches
    TestComputeEssentialMatrixCompute(container, True)

    # scaleFactor = config["ScaleFactor"]
    # DisplayImages(container.grayList, (scaleFactor, scaleFactor))