import cv2 as cv
import numpy as np

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


if __name__ == "__main__":
    
    # Initializing the container
    container = BundleAdjustmentContainer(config)
    
    # Loading the images
    container.imgList, container.grayList = LoadImages("data/", end=5)

    # Preprocessing the images
    container.grayList = PreprocessImages(container.grayList)

    # Testing keypoints
    TestFeatureExtraction(container, False)

    # Testing matches
    TestFeatureMatching(container, True)

    # scaleFactor = config["ScaleFactor"]
    # DisplayImages(container.grayList, (scaleFactor, scaleFactor))