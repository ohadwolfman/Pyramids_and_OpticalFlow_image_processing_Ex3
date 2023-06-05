import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316552496

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number!)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # Convert images to float64 data type
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    rows, cols = im1.shape
    half_win = win_size // 2
    points = []  # List to store original points
    deltas = []  # List to store displacement vectors (dU, dV)

    # Iterate over the image with the given step size
    for x in range(half_win, rows - half_win, step_size):
        for y in range(half_win, cols - half_win, step_size):
            # Extract the window from both images
            window1 = im1[x - half_win: x + half_win + 1, y - half_win: y + half_win + 1]
            window2 = im2[x - half_win: x + half_win + 1, y - half_win: y + half_win + 1]

            # Compute gradients using Sobel operators
            I_x = cv2.Sobel(src=window1, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)  # gradient x
            I_y = cv2.Sobel(src=window1, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  # gradient y

            # Stack gradients to form the matrix A and compute the vector b
            A = np.column_stack((I_x.flatten(), I_y.flatten()))
            b = (window1 - window2).flatten()

            # Solve the linear system using least squares
            flow_vector, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Append the original point and the displacement vector to the lists
            points.append([y, x])
            deltas.append([flow_vector[0], flow_vector[1]])

    # Convert lists to numpy arrays and return
    return np.array(points), np.array(deltas)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Define parameters for the Lucas-Kanade optical flow algorithm
    lk_params = dict(winSize=(winSize, winSize),
                     maxLevel=k,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create an array to store the optical flow vectors
    flow_vectors = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)

    # Create a grid of points using the step size
    points = np.mgrid[0:gray1.shape[0]:stepSize, 0:gray1.shape[1]:stepSize].T.reshape(-1, 1, 2).astype(np.float32)

    # Compute the optical flow for each point
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None, **lk_params)

    # Update the flow vectors with the computed displacements
    flow_vectors[points[:, 0, 1].astype(int), points[:, 0, 0].astype(int)] = new_points[:, 0, :] - points[:, 0, :]

    return flow_vectors


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass

