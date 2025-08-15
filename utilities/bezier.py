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

def curvature_scalar(prime, second):
    """
    Compute the curvature of a line segment defined by two points p0 and p1.
    
    Parameters:
    - p0: First point (numpy array).
    - p1: Second point (numpy array).
    - accurate: If True, uses a more accurate method for curvature calculation.
    
    Returns:
    - Curvature value as a float.
    """
    """ cross1 = np.cross(prime_f0, second_f0)
    cross2 = np.cross(prime_f1, second_f1)
    curvature1 = np.linalg.norm(cross1) / (np.linalg.norm(prime_f0)**3)
    curvature2 = np.linalg.norm(cross2) / (np.linalg.norm(prime_f1)**3)
    return (curvature1 + curvature2) / 2 """

    curvature1 = np.abs(second) / (1 + prime**2)**(3/2)
    return curvature1 
    

def unit(vector):
    """
    Normalize a vector to unit length.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def control_points(func, p0, p3, error_on_curvature=0.001):
    """
    Calculate control points for a cubic Bézier curve that approximates
    a given function y = f(x) between p0 and p3.

    Parameters:
    - func: function f(x)
    - p0, p3: start and end points as (x, y)
    - error_on_curvature: curvature threshold for linear approximation

    Returns:
    - p1, p2: control points
    """
    dt = 1e-5
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    x0, x1 = p0[0], p3[0]

    # First derivatives
    prime_f0 = (func(x0 + dt) - func(x0 - dt)) / (2 * dt)
    prime_f1 = (func(x1 + dt) - func(x1 - dt)) / (2 * dt)

    # Second derivatives
    second_f0 = (func(x0 + dt) - 2 * func(x0) + func(x0 - dt)) / (dt**2)
    second_f1 = (func(x1 + dt) - 2 * func(x1) + func(x1 - dt)) / (dt**2)

    # Tangent unit vectors
    T0 = unit(np.array([1.0, prime_f0], dtype=float))
    T1 = unit(np.array([1.0, prime_f1], dtype=float))

    # Average curvature at endpoints
    kappa_avg = 0.5 * (
        curvature_scalar(prime_f0, second_f0) +
        curvature_scalar(prime_f1, second_f1)
    )

    # Control point distance along tangents
    if kappa_avg <= error_on_curvature:
        # Nearly straight: 1/3 of chord length
        dist = np.linalg.norm(p3 - p0) / 3.0
    else:
        R = 1.0 / kappa_avg  # radius of curvature
        cos_theta = np.clip(np.dot(T0, T1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        dist = (4 * R * np.tan(theta / 4.0)) / 3.0

    # Control points
    p1 = p0 + T0 * dist
    p2 = p3 - T1 * dist  # minus so it points toward p0

    return p1, p2


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
