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
        if 0 <= px < ans.shape[0] and 0 <= py < ans.shape[1]:
            ans[px, py, 0] = deltas[index][0]
            ans[px, py, 1] = deltas[index][1]
    return ans


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def opticalFlowCV2(im1: np.ndarray, im2: np.ndarray, block_size=11,
                max_corners=5000, quality_level=0.00001, min_distance=1) -> (np.ndarray, np.ndarray):
    """
    Calculates the optical flow between two images using the Lucas-Kanade method.
    :param im1: First image
    :param im2: Second image
    :param block_size: Size of the block for corner detection
    :param max_corners: Maximum number of corners to detect
    :param quality_level: Minimum accepted quality of corners
    :param min_distance: Minimum distance between corners
    :return: Optical flow movements and the best detected corner points
    """
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')

    # Use the goodFeaturesToTrack function to find the best corner points
    corners = cv2.goodFeaturesToTrack(im1, max_corners, quality_level, min_distance, blockSize=block_size)

    # Calculate optical flow using the calcOpticalFlowPyrLK function
    movements, _, _ = cv2.calcOpticalFlowPyrLK(im1, im2, corners, None)

    return movements, corners

def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    movements, _ = opticalFlowCV2(im1, im2)  # Calculate optical flow using Lucas-Kanade method

    mean_movement = np.mean(movements, axis=0)  # Compute the mean movement in x and y directions

    translation_matrix = np.array([[1, 0, mean_movement[0, 0]],
                                   [0, 1, mean_movement[0, 1]],
                                   [0, 0, 1]])  # Construct the translation matrix

    return translation_matrix


# def rigidlkdemo(img_path):
#     print("---------------------------------------------------------------------------")
#     print("Rigid lk demo")
#     img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
#     angle = 0.8
#
#     rigid = np.array([[np.cos(angle), -np.sin(angle), -1],
#                       [np.sin(angle), np.cos(angle), -1],
#                       [0, 0, 1]], dtype=np.float32)
#
#     img_2 = cv2.warpPerspective(img_1, rigid, img_1.shape[::-1])
#     cv2.imwrite('input/imRigidA1.jpg', img_1)
#     cv2.imwrite('input/imRigidA2.jpg', img_2)
#
#     f, ax = plt.subplots(1, 2)
#     ax[0].set_title('CV Rigid')
#     ax[0].imshow(img_2, cmap='gray')
#
#     start = time.time()
#     my_rigid = findRigidLK(img_1, img_2)
#     end = time.time()
#     print("Time: {:.2f}".format(end - start))
#     print("My Rigid Matrix:\n", my_rigid, "\n\nOriginal Rigid Matrix:\n", rigid)
#
#     my_warp = warpImage(img_2, my_rigid)
#     ax[1].set_title('My Rigid')
#     ax[1].imshow(my_warp, cmap='gray')
#     plt.show()


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    # Perform 2D cross-correlation between the images
    correlation = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im1) * np.conj(np.fft.fft2(im2))))

    # Find the peak correlation location
    peak_loc = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Compute the translation vector
    translation_vector = np.array([peak_loc[1] - im1.shape[1] / 2, peak_loc[0] - im1.shape[0] / 2])

    # Construct the translation matrix
    translation_matrix = np.array([[1, 0, translation_vector[0]],
                                   [0, 1, translation_vector[1]],
                                   [0, 0, 1]])

    return translation_matrix


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    # Perform 2D cross-correlation between the images
    correlation = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im1) * np.conj(np.fft.fft2(im2))))

    # Find the peak correlation location
    peak_loc = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Compute the translation vector
    translation_vector = np.array([peak_loc[1] - im1.shape[1] / 2, peak_loc[0] - im1.shape[0] / 2])

    # Compute the rotation angle
    angle = np.angle(correlation[peak_loc])

    # Construct the rigid matrix
    rigid_matrix = np.array([[np.cos(angle), -np.sin(angle), translation_vector[0]],
                             [np.sin(angle), np.cos(angle), translation_vector[1]],
                             [0, 0, 1]])

    return rigid_matrix


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Warps image 2 according to the transformation matrix T and displays both image 1 and the wrapped version of image 2.
    :param im1: Input image 1 in grayscale format.
    :param im2: Input image 2 in grayscale format.
    :param T: Transformation matrix such that each pixel in image 2 is mapped under homogeneous coordinates to image 1 (p2 = T*p1).
    :return: Wrapped version of image 2.
    """
    im2_warped = cv2.warpPerspective(im2, T, (im1.shape[1], im1.shape[0]))
    stacked_images = np.hstack((im1, im2_warped))
    cv2.imshow("Image Comparison", stacked_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im2_warped


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
    pyramid = [img]  # List to store pyramid levels

    for _ in range(levels - 1):
        img = cv2.pyrDown(img)  # Downsample the image
        pyramid.append(img)

    return pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    # create answer array
    answer = []

    # find gaussian pyramid
    img_list = gaussianPyr(img, levels)

    # add images to answer
    for i in range(levels):
        curr = img_list[i]
        answer.append(curr - cv2.blur(curr, (5, 5)))

    return answer


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    # Start with the last image in the laplacian pyramid
    answer = lap_pyr[-1]

    # Add images to restore the original image
    for img in range(len(lap_pyr) - 2, -1, -1):
        expanded_img = cv2.pyrUp(answer)  # Upsample the previous image
        expanded_img = cv2.resize(expanded_img, lap_pyr[img].shape[:2][::-1])  # Resize to match current image size
        answer = expanded_img + lap_pyr[img]  # Add the current laplacian image

    # Normalize pixel values to [0, 1]
    answer = cv2.normalize(answer, None, 0, 1, cv2.NORM_MINMAX)
    return answer


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
    # find blended image
    lapA = laplaceianReduce(img_1, levels)
    lapB = laplaceianReduce(img_2, levels)
    gaussPyr = gaussianPyr(mask, levels)

    # create answer array
    lapC = []

    # for every level in the pyramids
    for k in range(levels):
        img = gaussPyr[k] * lapA[k] + (1 - gaussPyr[k]) * lapB[k]
        lapC.append(img)

    # Reconstruct all levels to get blended image
    blended = laplaceianExpand(lapC)
    blended = np.resize(blended, [679, 1023, 3])

    # find naive blend
    naive = img_1 * mask + img_2 * (1 - mask)

    return np.array(naive), np.array(blended)

