"""CV core of the DLO direction pipeline: diff image -> fitted line params.

Single source of truth for ``get_line_params_from_diff`` and its helper closure
(threshold + neighbor cleanup + component filter + border mask + L1/Huber line
fit), shared by the online ``DirectionExtractor`` model. Depends only on cv2 and
numpy so it imports without pulling h5py or any offline preprocessing driver.
"""

import cv2
import numpy as np


def apply_neighbor_rule(bw_img):
    """
    Applies the modified neighbor rule.
    """
    img_binary = (bw_img > 127).astype(np.uint8)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count_ones = cv2.filter2D(img_binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    output_binary = img_binary.copy()

    # Rule 1: Pixel is White (1), switch to Black (0) if < 4 neighbors are White
    to_black = (img_binary == 1) & (neighbor_count_ones < 4)

    # Rule 2: Pixel is Black (0), switch to White (1) if > 4 neighbors are White
    to_white = (img_binary == 0) & (neighbor_count_ones > 4)

    output_binary[to_black] = 0
    output_binary[to_white] = 1

    return (output_binary * 255).astype(np.uint8)

def remove_small_black_components(bw_img, min_size=30):
    """
    Finds black connected components. Turns those smaller than min_size to white.
    """
    inv_bw = cv2.bitwise_not(bw_img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_bw, connectivity=8)
    output_img = bw_img.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            output_img[labels == i] = 255

    return output_img

def get_black_points(img):
    """Returns (x, y) coordinates of all black pixels."""
    ys, xs = np.where(img == 0)
    if len(xs) == 0:
        return None
    return np.column_stack((xs, ys))


def mask_borders_white(img):
    """
    Sets a 10% rectangular border (annulus) around the image to white (255).
    """
    # Get dimensions
    h, w = img.shape[:2]

    # Calculate 10% margins
    margin_h = int(h * 0.1)
    margin_w = int(w * 0.1)

    # Create a copy to avoid modifying the original
    new_img = img.copy()

    # Set Top and Bottom rows to White
    # The ... handles both 2D (grayscale) and 3D (color) arrays
    new_img[:margin_h, ...] = 255
    new_img[-margin_h:, ...] = 255

    # Set Left and Right columns to White
    new_img[:, :margin_w, ...] = 255
    new_img[:, -margin_w:, ...] = 255

    return new_img

def fit_line_l1(points):
    """
    Fits a line using L1 norm. Returns (vx, vy, x0, y0).
    """
    if points is None or len(points) < 2:
        return np.zeros(4, dtype=np.float32) # Return zeros if no line found

    # line_params = cv2.fitLine(points, cv2.DIST_L1, 0, 0.01, 0.01)
    line_params = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
    # line_params = cv2.fitLine(points, cv2.DIST_WELSCH, 0, 0.01, 0.01)
    return line_params.flatten() # Returns [vx, vy, x0, y0]

def get_line_params_from_diff(diff_rgb_float: np.ndarray) -> np.ndarray:
    """
    Extract line parameters from a background-subtracted diff image.

    This is the reusable core of the CV pipeline, intended to be called both
    from the offline H5 processor and from the online DirectionExtractor model.

    Args:
        diff_rgb_float: (H, W, 3) float32 diff image (may contain negatives).

    Returns:
        np.ndarray of shape (4,): [vx, vy, x0, y0], or zeros if no line found.
    """
    abs_img = np.clip(np.abs(diff_rgb_float), 0.0, 1.0)
    uint8_img = (abs_img * 255).astype(np.uint8)

    gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    bw = apply_neighbor_rule(bw)
    bw = remove_small_black_components(bw, min_size=75)
    bw = mask_borders_white(bw)
    points = get_black_points(bw)
    return fit_line_l1(points)
