def find_leftmost_point(points: list[tuple]):
    leftmost_point = points[0]
    for point in points[1:]:
        if point[0] < leftmost_point[0]:
            leftmost_point = point
        elif point[0] == leftmost_point[0] and point[1] > leftmost_point[1]:
            leftmost_point = point
    return leftmost_point

def orientation(p: tuple, q: tuple, r: tuple):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def convex_hull(points: list[tuple]):
    n = len(points)
    if n < 3:
        return []
    hull = []
    l = find_leftmost_point(points)
    p = l
    q = None
    while True:
        hull.append(p)
        q = points[0]
        for r in points[1:]:
            o = orientation(p, q, r)
            if o == 2 or (o == 0 and ((q[0] - p[0])**2 + (q[1] - p[1])**2) < ((r[0] - p[0])**2 + (r[1] - p[1])**2)):
                q = r
        p = q
        if p == l:
            break
    return hull