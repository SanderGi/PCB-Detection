import cv2
import numpy as np

def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def distance_to_line(pt, line):
    (x1, y1) = pt
    (x0, y0, a, b) = line
    return abs(a * (y0 - y1) - b * (x0 - x1))

def distance_to_line_pts(pt, line):
    (x, y) = pt
    (x1, y1), (x2, y2) = line
    return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

def preprocess(image):
    """Convert to grayscale and apply bilateral filter to blur while preserving edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    gray = (255 * gray).astype(np.uint8)
    bi = cv2.bilateralFilter(gray, 5, 75, 75)
    return bi

def find_corner_candidates(preprocessed):
    """Find corner candidates using Harris corner detection."""
    dst = cv2.cornerHarris(preprocessed, 2, 3, 0.04)
    mask = np.zeros_like(preprocessed)
    mask[dst>0.01*dst.max()] = 255
    coordinates = np.argwhere(mask)
    coordinates = [tuple(pt.tolist()) for pt in coordinates]
    coordinates = [(x, y) for y, x in coordinates]
    return coordinates

def find_corners(image):
    """Finds the pcb corners in the image in clockwise order starting with the top left."""
    preprocessed = preprocess(image)
    coordinates = find_corner_candidates(preprocessed)
    height, width = preprocessed.shape

    # find the coordinates closest to the corners of the image (in clockwise order)
    c1 = sorted(coordinates, key=lambda x: distance(x, (0, 0)))[0]
    c2 = sorted(coordinates, key=lambda x: distance(x, (width, 0)))[0]
    c3 = sorted(coordinates, key=lambda x: distance(x, (width, height)))[0]
    c4 = sorted(coordinates, key=lambda x: distance(x, (0, height)))[0]

    # find the coordinates closest to the edges of the image
    samp = 30
    top = sorted(coordinates, key=lambda x: distance_to_line(x, (0, 0, 1, 0)))[:samp]
    bottom = sorted(coordinates, key=lambda x: distance_to_line(x, (0, height, 1, 0)))[:samp]
    left = sorted(coordinates, key=lambda x: distance_to_line(x, (0, 0, 0, 1)))[:samp]
    right = sorted(coordinates, key=lambda x: distance_to_line(x, (width, 0, 0, 1)))[:samp]
    edge_lines = [top, right, bottom, left]
    # average the coordinates
    average_edge_coords = [tuple(np.mean(e, axis=0)) for e in edge_lines]
    # find the best fit line
    lines = []
    for origin, endpoints, points in zip(average_edge_coords, [(c1, c2), (c2, c3), (c3, c4), (c4, c1)], edge_lines):
        candidates = []
        for pt in endpoints:
            candidates.append((origin, pt))
        best_candidate = min(candidates, key=lambda x: np.sum([distance_to_line_pts(pt, x) for pt in points]))
        lines.append(best_candidate)
    # find line intersections
    true_corner_coords = []
    for i in range(len(lines)):
        (x1, y1), (x2, y2) = lines[i]
        (x3, y3), (x4, y4) = lines[(i + 1) % len(lines)]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        x /= denom
        y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        y /= denom
        true_corner_coords.append((x, y))
    true_corner_coords = true_corner_coords[-1:] + true_corner_coords[:-1]

    return true_corner_coords

def unwarp(image):
    """Unwarp the image, make the pcb the entire image."""
    h, w = image.shape[:2]
    corners = find_corners(image)
    src = np.array(corners, dtype=np.float32)
    dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped
