"""Even-odd polygon filling at original image pixel centers."""
import math
import numpy as np


def lasso_mask(shape, points):
    height, width = shape
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not 3 <= len(coordinates) <= 2000:
        raise ValueError('Draw a closed region with at least three points (maximum 2000).')
    if not np.isfinite(coordinates).all():
        raise ValueError('Lasso coordinates must be finite.')
    if (np.any(coordinates < 0) or np.any(coordinates[:, 0] > width - 1)
            or np.any(coordinates[:, 1] > height - 1)):
        raise ValueError('Lasso coordinates are outside the image.')
    mask = np.zeros(shape, dtype=bool)
    first = max(0, math.ceil(float(coordinates[:, 1].min())))
    last = min(height - 1, math.floor(float(coordinates[:, 1].max())))
    for y in range(first, last + 1):
        crossings = []
        for i, (ax, ay) in enumerate(coordinates):
            bx, by = coordinates[(i + 1) % len(coordinates)]
            if (ay <= y < by) or (by <= y < ay):
                crossings.append(float(ax + (y - ay) * (bx - ax) / (by - ay)))
        crossings.sort()
        for left, right in zip(crossings[::2], crossings[1::2]):
            start, end = max(0, math.ceil(left)), min(width, math.ceil(right))
            mask[y, start:end] = True
    if not mask.any():
        raise ValueError('The lasso does not enclose any image pixels. Draw a larger region.')
    return mask
