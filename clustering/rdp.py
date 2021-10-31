from math import sqrt

def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def point_line_distance(point, start, end):
    if start[0] == end[0] and start[1] == end[1] and start[2] == end[2]:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) * (start[2] - point[2]) -
            (start[0] - point[0]) * (end[1] - start[1]) * (end[2] - start[2])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 + (end[2] - start[2]) ** 2
        )
        return n / d

def rdp_with_index(points, indices, epsilon):
    """rdp with returned point indices
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        first_points, first_indices = rdp_with_index(points[:index+1], indices[:index+1], epsilon)
        second_points, second_indices = rdp_with_index(points[index:], indices[index:], epsilon)
        results = first_points[:-1] + second_points
        results_indices = first_indices[:-1] + second_indices
    else:
        results, results_indices = [points[0], points[-1]], [indices[0], indices[-1]]
    return results, results_indices