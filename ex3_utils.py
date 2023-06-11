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
    # Luckas kanade:
    # The function f(x+u, y+u) we want is equivalent to: f(x,y) + (df/dx)*u + (df/dy)*v
    # We need to find the u,v that minimizes:
    # E(u,v) = Sigma((I1 + Ix*u + Iy*v) - I2)^2
    #        = Sigma(It + Ix*u + Iy*v)^2     'I' is the gradient
    #   I1 is f(x,y),   Ix is (df/dx),   Iy is (df/dy) -I2

    # Convert images to float64 data type
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    rows, cols = im1.shape
    half_win = win_size // 2

    # The kernel for derivative axes
    deriv = np.array([[-1,0,1]])

    # Calculating the gradients
    Ix = cv2.filter2D(im2, -1, deriv, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, deriv.T, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    # We want to minimize this: (It + Ix*u + Iy*v)
    # => It + Ix*u + Iy*v = 0 =>  Ix*u + Iy*v = -It
    #    | Ix[1]   Iy[1]  |              | It[1] |
    # => | ...     ...    | * | u | =  -   ...          => Av = b
    #    | Ix[n]   Iy[n]] |   | v |      | It[n] |

    # [u v] = (A.T * A)^-1 * A.T * b
    points = []  # List to store original points
    deltas = []  # List to store displacement vectors (dU, dV)

    # Iterate over the image with the given step size
    for x in range(half_win, rows - half_win + 1, step_size):
        for y in range(half_win, cols - half_win + 1, step_size):
            # Extract the window from both images
            window1 = Ix[x - half_win: x + half_win + 1, y - half_win: y + half_win + 1]
            window2 = Iy[x - half_win: x + half_win + 1, y - half_win: y + half_win + 1]
            window3 = It[x - half_win: x + half_win + 1, y - half_win: y + half_win + 1]

            # Stack gradients to form the matrix A and compute the vector b
            A = np.vstack((window1.flatten(), window2.flatten())).T  # A = [Ix, Iy]
            AT_A = np.dot(A.T, A)

            eigen_values = np.linalg.eigvals(AT_A)
            if eigen_values.min() <= 1 or eigen_values.max() / eigen_values.min() >= 100:
                continue

            AT_b = np.dot(A.T, (-1 * window3.flatten()).T).reshape(2, 1)
            u_v = np.dot(np.linalg.inv(AT_A), AT_b)

            # Append the original point and the displacement vector to the lists
            points.append([y,x])
            deltas.append([u_v[0], u_v[1]])

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
    # if the image is RGB -> convert to gray
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # If the images have different shapes
    if img1.shape != img2.shape:
        raise Exception("The images must be in the same size")

    # Generate 2 lists of gaussian pyramids to img1 and img2
    # The reverse is for the for loop below, the pyramids list go from bigger to smaller, and we want the opposite
    pyramids1 = (gaussianPyr(img1, k))
    pyramids1.reverse()
    pyramids2 = gaussianPyr(img2, k)
    pyramids2.reverse()

    # Generate the optical flow of the 2 smallest images
    points, deltas = opticalFlow(pyramids1[0], pyramids2[0], stepSize, winSize)

    # Increasing to the largest image
    for curr_img in range(1, k):
        curr_points, curr_deltas = opticalFlow(pyramids1[curr_img], pyramids2[curr_img], stepSize, winSize)

        # Fixing the values of points, deltas to one larger image: 2*(u v)
        for j in range(len(points)):
            points[j] = [element * 2 for element in points[j]]
            deltas[j] = [element * 2 for element in deltas[j]]

        for pixel, u_v in zip(curr_points, curr_deltas):
            if pixel not in points:
                points = np.vstack([points, pixel])  # 'vstack' is similar to concat
                deltas = np.concatenate([deltas, np.expand_dims(u_v, axis=0)], axis=0)

            else:
                # Add the current vector u_v to the existing vector for the pixel
                idx = np.where((points[:, 0] == pixel[0]) & (points[:, 1] == pixel[1]))[0]
                if len(idx) > 0:
                    idx = idx[0]
                    # Add the current vector u_v to the existing vector for the pixel
                    deltas[idx] += u_v

    # Creating an array for the final result with (m, n, 2) shape
    ans = np.zeros((img1.shape[0], img1.shape[1], 2))

    for index in range(len(points)):
        px = points[index][1]
        py = points[index][0]
        # if 0 <= px < ans.shape[0] and 0 <= py < ans.shape[1]:
        #     ans[px, py, 0] = deltas[index][0]
        #     ans[px, py, 1] = deltas[index][1]
        ans[px][py][0] = deltas[index][0]
        ans[px][py][1] = deltas[index][1]
    return ans


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
    gaussianPyrList = [img]
    for i in range(0, levels):
        lower_img = cv2.pyrDown(gaussianPyrList[i - 1])
        gaussianPyrList.append(lower_img)
    return gaussianPyrList


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

