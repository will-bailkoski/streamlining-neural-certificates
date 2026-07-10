import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull


def split_simplex_middle(vertices: np.ndarray) -> np.ndarray:
    "returns the centroid and midpoints of each edge to be added to a mesh"

    pairs = combinations(range(vertices.shape[0]), 2)

    centroid = np.mean(vertices, axis=0)
    midpoints = [(vertices[i] + vertices[j]) / 2.0 for i, j in pairs]

    return np.vstack([centroid, *midpoints])


# def split_circumcircle(center: np.ndarray, hull: ConvexHull):


#     return center
