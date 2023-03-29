def parametric_ease(t):
    sqt = t * t
    return sqt / (2 * (sqt - t) + 1)


def bezier_ease(t):
    return t * t * (3 - 2 * t)


def weighted_average(t1, t2, w):
    return tuple(x * (1 - w) + y * w for x, y in zip(t1, t2))
