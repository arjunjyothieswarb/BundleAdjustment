import cv2 as cv
import numpy as np
import yaml
import os
import gtsam.symbol_shorthand as gtsam_symbol

import log_utils as log

def LoadImages(dirPath: str, start: float=0, end: float=np.inf) -> tuple[list[np.ndarray]]:
    """
    Loads all image files from the specified directory and returns them as a list of numpy arrays.
    Args:
        dirPath (str): The path to the directory containing image files.
        start (float): The number of initial images to skip.[Default value = 0]
    Returns:
        imageList: A list of color images loaded as numpy arrays.
        grayList: A list of grayscale images loaded as numpy arrays.
    Notes:
        - Supported image formats are .tif, .jpg, and .png.
        - Images are read in sorted order by filename.
        - Requires the 'os' and 'cv2' (as 'cv') modules to be imported.
    """

    imageList = []
    grayList = []
    
    # Getting all the image files in the directory
    fileNames = [file for file in os.listdir(dirPath) if file.endswith((".tif", ".jpg", ".png"))]
    
    # Reading all the images in order
    count = 0
    for file in sorted(fileNames):
        filePath = os.path.join(dirPath, file)
        image = cv.imread(filePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if image is not None:
            count = count + 1
            if count < start:
                continue
            if count > end:
                break
            imageList.append(image)
            grayList.append(gray)
        else:            
            log.warn(f"Unable to open {file}! Ignoring {file}")
            continue

    if len(imageList) < 1:
        log.error("No supported images found in the directory!")
        exit()

    return imageList, grayList

def DisplayImages(imageList: list[np.ndarray], scale: float) -> None:
    """
    Displays a list of images in separate windows.
    Args:
        imageList (list[np.ndarray]): A list of images represented as NumPy arrays to be displayed.
    Returns:
        None
    Each image in the list is displayed in a separate window with a unique title.
    The function waits for a key press before closing each window and proceeding to the next image.
    """
    
    count = 1
    for image in imageList:
        displayString = f"Image {count}"
        log.info(f"Displaying {displayString}.")
        
        if image is not None:
            # Scaling the output image
            if scale is not None:
                image = cv.resize(image, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_LINEAR)
            
            # Displaying the image, close window on key press
            cv.imshow(displayString, image)
            cv.waitKey()
            cv.destroyAllWindows()
        
        else:
            continue

        count += 1

class BundleAdjustmentContainer():
    
    def __init__(self, config) -> None:
        
        # Storing the directory path
        dirPath = config["ImageDir"]
        self.dirPath = os.path.join(os.getcwd(), dirPath)

        # Initializing the camera matrix
        self.CAM_MATRIX = np.array([
                            [1662,    0, 540],
                            [   0, 1673, 960],
                            [   0,    0,   1]], dtype=np.float32)

        # Loading SIFT params
        self.siftParams = {
            "nFeatures": config["SIFT"]["nFeatures"],
            "nOctaveLayers": config["SIFT"]["nOctaveLayers"],
            "contrastThreshold": config["SIFT"]["contrastThreshold"],
            "edgeThreshold": config["SIFT"]["edgeThreshold"],
            "sigma": config["SIFT"]["sigma"]
        }

        self.lowes_const = config["FeatureMatching"]["lowes_const"]
        self.RANSAC_THRESH = config["FeatureMatching"]["RANSAC_THRESH"]
        
        # Setting the minimum number of matching key-points
        self.MIN_MATCH_CNT = config["FeatureMatching"]["MIN_MATCH_COUNT"]

        # Initializing image list
        self.imgList = []
        self.grayList = []

        # Initializing the key-point list and the descriptor list
        self.kpList = []
        self.kpIDList = []
        self.desList = []

        # Initializing the matcher
        self.BF = cv.BFMatcher()

        # Initalizing item counter
        self.itemCounter = {'X': 0, 'L': 0} # X is reserved for Camera poses, Y is reserved for Landmarks

    def __len__(self):
        return len(self.kpList)


    def extractFeatures(self, imageList=None) -> tuple[list, list, list]:
        """
        Extracts SIFT keypoints and descriptors from each image in the image list.
        
        Args:
            imageList (list): List of images.

        Returns:
            tuple[list, list]: A tuple containing two lists:
                - kpList: A list of keypoints for each image.
                - desList: A list of descriptors for each image.
                - kpIDList: A list of keypoint IDs
        """

        if imageList == None:
            imageList = self.grayList

        # Creating the SIFT object
        sift = cv.SIFT.create(
            nfeatures = self.siftParams["nFeatures"],
            nOctaveLayers = self.siftParams["nOctaveLayers"],
            contrastThreshold = self.siftParams["contrastThreshold"],
            edgeThreshold = self.siftParams["edgeThreshold"]
        )

        kpList = []
        kpIDList = []
        desList = []

        # Detecting and computing features and descriptors
        for image in imageList:
            
            kp, des = sift.detectAndCompute(image, None)
            
            # Creating a list of IDs corresponding to each kp - Initialzed as None
            kpID = [None] * len(kp)

            # Appending them to a list
            kpList.append(kp)
            desList.append(des)
            kpIDList.append(kpID)

        return (kpList, desList, kpIDList)
    

    def findMatches(self, idx1: int, idx2: int) -> list:
        """
        Finds matching key-points between images of the indices provided. Returns
        the good matches.

        Args:
            idx1 (int): Index of image 1
            idx2 (int): Index of image 2

        Returns:
            list: List of matches
        """

        # BF matcher with default params
        bf = cv.BFMatcher()
        matchesList = []

        matches = bf.knnMatch(self.desList[idx1], self.desList[idx2], k=2)

        # Applying Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < self.lowes_const*n.distance:
                good.append(m)
        
        return good
    
    def computeEssentialMatrix(self, idx1: int, idx2: int) -> tuple[np.array, int]:
        """
        Computes the essential matrix between 2 cameras, given the indices of the images.

        Args:
            idx1 (int): Index of image 1
            idx2 (int): Index of image 2

        Returns:
            tuple[np.array, int]: A tuple containing the essential matrix and number of inlier matches
        """
        # Getting the key-points
        kp1 = self.kpList[idx1]
        kp2 = self.kpList[idx2]

        # Computing the matches
        matches = self.findMatches(idx1, idx2)

        # Checking for min number of matches
        if len(matches) < self.MIN_MATCH_CNT:
            log.warn(f"Not enough matches between image {idx1} and image {idx2}!")
            return[None, -1]
        
        # Extracting source pts and destination pts
        srcPts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        dstPts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

        # Computing the essential matrix
        E, mask = cv.findEssentialMat(srcPts, dstPts, cameraMatrix=self.CAM_MATRIX, method=cv.RANSAC, threshold=self.RANSAC_THRESH)

        mask = mask.ravel().tolist()
        # numMatches = mask.count(1)

        # Get the kpIds
        kpID1 = self.kpIDList[idx1]
        kpID2 = self.kpIDList[idx2]

        numMatches = 0
        for idx, mask_element in enumerate(mask):
            if not mask_element:
                continue
            m = matches[idx]

            # Assigning ID to landmarks
            kpID1[m.queryIdx] = gtsam_symbol.L(self.itemCounter['L'])
            kpID2[m.trainIdx] = gtsam_symbol.L(self.itemCounter['L'])

            # Updating counters
            self.itemCounter['L'] += 1
            numMatches += 1
            

        log.info(f"Num matches between images {idx1} and {idx2}: {numMatches}")

        return E, numMatches