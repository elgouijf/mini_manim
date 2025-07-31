import numpy as np
from math import comb

def bezier_decasteljeau(points, t):
    """
        Compute a point on a Bézier curve using De Casteljau's recursive algorithm.

    Parameters:
    - points: list or array of control points (each point should be a NumPy array).
    - t: parameter between 0 and 1 indicating the interpolation position.

    Returns:
    - A single point on the Bézier curve at parameter t.
    """
    n = len(points)
    if n == 1:
        return points[0]
    elif n == 2:
        return bezier_quadratic(t, *(points))
    elif n == 3:
        return bezier_cubic(t, *(points))
    else:
        new_points = []
        for i in range(n- 1):
            # Linearly interpolate between consecutive pairs of points
            new_points.append(points[i]*(1 - t) + points[i + 1]*t)

        return bezier_decasteljeau(new_points, t)


def bezier_closed(points, t):
    """
    Evaluate a Bézier curve at parameter t using the closed-form Bernstein polynomial definition.

    Parameters:
    - points: list or array of control points (each point should be a NumPy array).
    - t: parameter between 0 and 1 indicating the interpolation position.

    Returns:
    - A single point on the Bézier curve at parameter t.

    Note:
    For 3 or 4 control points, the closed-form can be optimized using specific quadratic
    and cubic Bézier formulas, which are more efficient and numerically stable.
    """
    n = len(points)
    if n == 1:
        return points[0]
    elif n == 2:
        return bezier_quadratic(t, *(points))
    elif n == 3:
        return bezier_cubic(t, *(points))
    
    bezier_t = np.zeros_like(points[0], dtype=float) # since the shape could be either (n,2) or (n,3) we better use zeros_like or .shape to keep it general
    for i in range(n):
        bezier_t += comb(n - 1, i)*((1 - t)**(n - 1 - i))*(t**i)*points[i]
    return bezier_t


def bezier_cubic(t, p0, p1, p2, p3):
    """
    Cubic Bézier interpolation using 4 control points.

    This is a special-case optimization for Bézier curves of degree 3.
    It uses the expanded Bernstein polynomial form for faster computation.
    """

    return ((1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3)

def bezier_quadratic(t, p0, p1, p2):
    """
    Quadratic Bézier interpolation using 3 control points.

    This is a special-case optimization for Bézier curves of degree 2.
    It avoids recursion for better performance and readability.
    """
    return ((((1 - t)**2) * p0 )+ (2 * (1 - t) * t * p1) + (t**2) * p2)


def lagrange(x, points):
    """
    Compute a point on a polynomial interpolation curve using the Lagrange interpolation formula.

    Parameters:
    - points: list of known (x_i, y_i) tuples or arrays.
    - x: the x-value at which to evaluate the interpolating polynomial.

    Returns:
    - A tuple (x, y) representing the interpolated value at x.

    Note:
    This implementation assumes distinct x_i values in the input points.
    """
    n = len(points)
    poly = 0
    for i in range(n):
        Li_x = 1
        for j in range(n):
            if j != i :
                Li_x *= (x - points[j][0])/(points[i][0] - points[j][0])
        poly += Li_x*points[i][1]

    return (x, poly)
