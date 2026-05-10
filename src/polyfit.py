"""Polynomial fitting for lane curves.

Given a cloud of (x, y) lane pixels (from sliding window), fit a 2nd-degree
polynomial x = a*y^2 + b*y + c using least squares. The polynomial coefficients
[a, b, c] represent the smooth lane curve.

Why x = f(y) instead of y = f(x):
    Lanes are predominantly vertical in the bird's-eye view. For each row y,
    there is typically one x per lane. Fitting y = f(x) would fail on near-
    vertical sections (infinite slope).

Why 2nd-degree:
    Degree 1 (line) cannot model curves.
    Degree 2 (parabola) models gentle highway curvature — the standard choice.
    Degree 3+ overfits to pixel noise, producing wavy curves that don't match
    real road geometry.
"""

from __future__ import annotations

import numpy as np


DEFAULT_DEGREE = 2


def fit_lane_polynomial(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    degree: int = DEFAULT_DEGREE,
) -> np.ndarray | None:
    """Least-squares fit of x = f(y) through a lane pixel cloud.

    Args:
        x_pixels: 1D array of x-coordinates of lane pixels.
        y_pixels: 1D array of y-coordinates of lane pixels.
        degree:   polynomial degree (default 2).

    Returns:
        Coefficients array of length (degree + 1), highest power first.
        For degree 2: [a, b, c] meaning x = a*y^2 + b*y + c.
        Returns None if fewer than (degree + 1) points are available.
    """
    if len(x_pixels) < degree + 1:
        return None
    return np.polyfit(y_pixels, x_pixels, degree)


def evaluate_polynomial(
    coeffs: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Compute x positions for each y using polynomial coefficients.

    For degree-2 coeffs [a, b, c] and y_values:
        result[i] = a * y_values[i]**2 + b * y_values[i] + c
    """
    return np.polyval(coeffs, y_values)


def polynomial_curvature(
    coeffs: np.ndarray,
    y_eval: float,
) -> float:
    """Compute the radius of curvature of the polynomial at a given y.

    Uses the standard curvature formula for a curve x(y):
        R = (1 + (dx/dy)^2)^(3/2) / |d2x/dy2|

    For a 2nd-degree polynomial x = a*y^2 + b*y + c:
        dx/dy   = 2*a*y + b
        d2x/dy2 = 2*a

    Args:
        coeffs: degree-2 coefficients [a, b, c].
        y_eval: y-value at which to evaluate the curvature (in pixels).

    Returns:
        Radius of curvature in pixels. Larger = straighter road.
        Returns +inf for a perfectly straight line (a = 0).
    """
    if len(coeffs) != 3:
        raise ValueError("polynomial_curvature expects degree-2 coefficients")
    a, b, _ = coeffs
    if abs(a) < 1e-12:
        return float("inf")
    dx_dy = 2 * a * y_eval + b
    d2x_dy2 = 2 * a
    return ((1 + dx_dy ** 2) ** 1.5) / abs(d2x_dy2)


if __name__ == "__main__":
    # Self-test: fit a known parabola and check we recover the coefficients
    print("Testing polynomial fit on a synthetic parabola...")
    
    # Ground truth: x = 0.001 * y^2 - 0.5 * y + 200
    true_coeffs = np.array([0.001, -0.5, 200.0])
    y = np.linspace(100, 700, 100)
    x_clean = np.polyval(true_coeffs, y)
    
    # Add some noise (like a real lane pixel cloud would have)
    np.random.seed(42)
    x_noisy = x_clean + np.random.normal(0, 5, size=y.shape)
    
    # Fit and compare
    fitted = fit_lane_polynomial(x_noisy, y)
    print(f"True coefficients:   {true_coeffs}")
    print(f"Fitted coefficients: {fitted}")
    
    # Should match closely
    error = np.linalg.norm(true_coeffs - fitted)
    print(f"Coefficient error norm: {error:.4f}")
    
    # Test curvature
    R = polynomial_curvature(fitted, y_eval=400)
    print(f"Radius of curvature at y=400: {R:.1f} pixels")
    print()
    
    # Edge case: too few points
    none_result = fit_lane_polynomial(np.array([1.0]), np.array([1.0]))
    print(f"Fit with 1 point (should be None): {none_result}")
    print()
    print("Module polyfit.py loaded successfully.")