import cv2 as cv
import numpy as np
import open3d

from utils import *
import log_utils as log


# Loading the config file globally
with open("./config/config.yaml") as f:
    config = yaml.safe_load(f)



def PreprocessImages(imgList: list[np.array]) -> list[np.array]:
    """
    Function that takes in a list of images and preprocesses them
    """

    clahe = cv.createCLAHE(clipLimit=config["Preprocessing"]["CLAHE"]["clipLimit"]) # Initializing a CLAHE object
    kernel_size = config["Preprocessing"]["GaussianBlur"]["KernelSize"] # Kernel size for gaussian blurring
    
    processedImgList = []
    for img in imgList:
        processedImg = cv.GaussianBlur(img, (kernel_size, kernel_size), 0) # Blurring the image
        processedImg = clahe.apply(processedImg) # Applying clahe
        processedImgList.append(processedImg)

    return processedImgList


if __name__ == "__main__":

    # Initializing the container
    container = BundleAdjustmentContainer(config)

    # Loading the images
    log.info("Loading images")
    imgList, grayList = LoadImages(config["ImageDir"], end=2)
    
    if imgList == None:
        log.error(f"No supported files found at {container.dirPath}!")
        exit()
    
    log.info("Successfully loaded images")

    # Extracting features
    log.info("Extracting features")
    container.kpList, container.desList, container.kpIDList = container.extractFeatures(grayList)

    # Feature matching
    log.info("Matching features")
    for idx in range(len(container) - 1):
        
        E, numMatches = container.computeEssentialMatrix(idx, idx+1)