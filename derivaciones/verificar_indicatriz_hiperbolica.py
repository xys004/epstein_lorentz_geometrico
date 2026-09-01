"""Checks for the hyperbolic tangent-indicatrix extension.

The script verifies exact algebraic factors and adversarial numerical examples.
It does not replace the standard Riemannian length--distance theorem, the
hyperbolic isoperimetric theorem, or a literature-priority audit.
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np
import sympy as sp


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def gamma(beta: np.ndarray) -> float:
    beta_squared = float(np.dot(beta, beta))
    require(0.0 <= beta_squared < 1.0, "A subluminal velocity is required")
    return 1.0 / math.sqrt(1.0 - beta_squared)


def hyperbolic_distance(beta_0: np.ndarray, beta_1: np.ndarray) -> float:
    gamma_0 = gamma(beta_0)
    gamma_1 = gamma(beta_1)
    gamma_relative = gamma_0 * gamma_1 * (
        1.0 - float(np.dot(beta_0, beta_1))
    )
    require(gamma_relative >= 1.0 - 2.0e-13, "Relative gamma fell below one")
    return math.acosh(max(1.0, gamma_relative))


def main() -> None:
    evidence: list[tuple[str, object]] = []

    # q'=q/D and |dF(V)|^2=|V|^2/D^2 imply exact invariance of
    # h=q^{-2}g_round.
    q, jacobian, vector_norm_squared = sp.symbols(
        "q D V2", positive=True
    )
    metric_residual = sp.simplify(
        (vector_norm_squared / jacobian**2) / (q / jacobian) ** 2
        - vector_norm_squared / q**2
    )
    require(metric_residual == 0, "The hyperbolic tangent metric is not invariant")
    evidence.append(("exact metric residual", metric_residual))

    # Along y=c*tau, |dT/dy|_round=kappa/q and hence
    # |dT/dy|_h=kappa/q^2=A/c^2.
    kappa, acceleration, c, d_tau = sp.symbols(
        "kappa A c d_tau", positive=True
    )
    dy_ds = q
    round_speed_with_respect_to_y = sp.simplify(kappa / dy_ds)
    require(
        round_speed_with_respect_to_y == kappa / q,
        "The chain rule from s to y lost a factor of q",
    )
    hyperbolic_speed = sp.simplify(round_speed_with_respect_to_y / q)
    require(hyperbolic_speed == kappa / q**2, "Wrong indicatrix speed")
    budget_element = sp.simplify(
        hyperbolic_speed.subs(kappa, acceleration * q**2 / c**2)
        * c
        * d_tau
    )
    require(budget_element == acceleration * d_tau / c, "Wrong budget element")
    evidence.append(("round speed dT/dy", round_speed_with_respect_to_y))
    evidence.append(("hyperbolic speed", hyperbolic_speed))
    evidence.append(("budget element", budget_element))

    # Endpoint distance on the future unit hyperboloid.
    rng = np.random.default_rng(20260805)
    for _ in range(300):
        beta_0 = rng.normal(size=3)
        beta_0 *= rng.uniform(0.0, 0.94) / np.linalg.norm(beta_0)
        beta_1 = rng.normal(size=3)
        beta_1 *= rng.uniform(0.0, 0.94) / np.linalg.norm(beta_1)

        gamma_0 = gamma(beta_0)
        gamma_1 = gamma(beta_1)
        four_velocity_0 = np.r_[gamma_0, gamma_0 * beta_0]
        four_velocity_1 = np.r_[gamma_1, gamma_1 * beta_1]
        minkowski_product = (
            four_velocity_0[0] * four_velocity_1[0]
            - float(np.dot(four_velocity_0[1:], four_velocity_1[1:]))
        )
        gamma_relative = gamma_0 * gamma_1 * (
            1.0 - float(np.dot(beta_0, beta_1))
        )
        require(
            abs(minkowski_product - gamma_relative) < 3.0e-13,
            "The relative-gamma formula is inconsistent",
        )
        require(
            abs(
                math.acosh(max(1.0, minkowski_product))
                - hyperbolic_distance(beta_0, beta_1)
            )
            < 3.0e-13,
            "The endpoint-distance formula is inconsistent",
        )
    evidence.append(("random endpoint formula", "300/300"))

    # Polygonal adversarial checks of length >= endpoint distance.
    minimum_margin = float("inf")
    for _ in range(600):
        points: list[np.ndarray] = []
        for _ in range(int(rng.integers(3, 10))):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            points.append(rng.uniform(0.0, 0.95) * direction)
        length = sum(
            hyperbolic_distance(points[index], points[index + 1])
            for index in range(len(points) - 1)
        )
        distance = hyperbolic_distance(points[0], points[-1])
        margin = length - distance
        minimum_margin = min(minimum_margin, margin)
        require(margin > -2.0e-12, "A polygon violated length >= distance")
    evidence.append(
        ("polygonal length >= distance", f"600/600; min margin={minimum_margin:.8g}")
    )

    # In 1+1, the budget is total variation of rapidity, not merely its net
    # change when the motion backtracks.
    monotone_rapidity = np.linspace(-1.3, 2.2, 401)
    backtracking_rapidity = np.array([-1.3, -0.2, 1.0, 0.1, 2.2])
    monotone_variation = float(np.abs(np.diff(monotone_rapidity)).sum())
    backtracking_variation = float(np.abs(np.diff(backtracking_rapidity)).sum())
    endpoint_change = abs(
        float(backtracking_rapidity[-1] - backtracking_rapidity[0])
    )
    require(abs(monotone_variation - 3.5) < 1.0e-12, "Monotone path did not saturate")
    require(
        backtracking_variation > endpoint_change + 1.0e-12,
        "Backtracking did not make the inequality strict",
    )
    evidence.append(
        (
            "rapidity total variation",
            f"monotone=3.5; reversed={backtracking_variation}; endpoint={endpoint_change}",
        )
    )

    # Hyperbolic polar metric: ds^2=dw^2+sinh(w)^2 dphi^2 has K=-1.
    rapidity = sp.symbols("w", positive=True)
    radial_factor = sp.sinh(rapidity)
    gaussian_curvature = sp.simplify(
        -sp.diff(radial_factor, rapidity, 2) / radial_factor
    )
    require(gaussian_curvature == -1, "Velocity space does not have K=-1")
    evidence.append(("Gaussian curvature", gaussian_curvature))

    # A hyperbolic circle is the exact equality case for the closed-loop
    # isoperimetric inequality and for uniform circular motion.
    beta = sp.tanh(rapidity)
    lorentz_factor = sp.cosh(rapidity)
    radius = sp.symbols("R", positive=True)
    circle_length = 2 * sp.pi * sp.sinh(rapidity)
    circle_area = 2 * sp.pi * (sp.cosh(rapidity) - 1)
    proper_acceleration = lorentz_factor**2 * beta**2 * c**2 / radius
    proper_period = 2 * sp.pi * radius / (lorentz_factor * beta * c)
    circle_budget = sp.trigsimp(proper_acceleration * proper_period / c)
    require(
        sp.trigsimp(circle_length - circle_budget) == 0,
        "Circular acceleration budget differs from indicatrix length",
    )
    isoperimetric_defect = sp.trigsimp(
        circle_length**2 - circle_area**2 - 4 * sp.pi * circle_area
    )
    require(isoperimetric_defect == 0, "A hyperbolic circle did not saturate")
    evidence.append(("circular budget", circle_budget))
    evidence.append(("circular Thomas angle magnitude", circle_area))
    evidence.append(("exact circle isoperimetric defect", isoperimetric_defect))

    # Numerical falsification attempt for smooth positive star-shaped loops in
    # H^2. This checks constants; the general theorem remains external.
    phi = np.linspace(0.0, 2.0 * math.pi, 20_000, endpoint=False)
    minimum_defect = float("inf")
    for _ in range(150):
        radial_offset = float(rng.uniform(0.35, 2.0))
        raw_coefficients = rng.normal(size=(5, 2))
        coefficient_scale = (
            float(rng.uniform(0.01, 0.16))
            * radial_offset
            / (np.sum(np.abs(raw_coefficients)) + 1.0e-15)
        )
        coefficients = raw_coefficients * coefficient_scale
        radial_coordinate = np.full_like(phi, radial_offset)
        radial_derivative = np.zeros_like(phi)
        for mode, (cosine_coefficient, sine_coefficient) in enumerate(
            coefficients, start=1
        ):
            radial_coordinate += cosine_coefficient * np.cos(mode * phi)
            radial_coordinate += sine_coefficient * np.sin(mode * phi)
            radial_derivative -= mode * cosine_coefficient * np.sin(mode * phi)
            radial_derivative += mode * sine_coefficient * np.cos(mode * phi)
        require(float(radial_coordinate.min()) > 0.0, "A test loop crossed the origin")

        perimeter = 2.0 * math.pi * float(
            np.mean(
                np.sqrt(
                    radial_derivative**2 + np.sinh(radial_coordinate) ** 2
                )
            )
        )
        area = 2.0 * math.pi * float(np.mean(np.cosh(radial_coordinate) - 1.0))
        defect = perimeter**2 - area**2 - 4.0 * math.pi * area
        minimum_defect = min(minimum_defect, defect)
        require(defect > -5.0e-8, "A test loop violated the isoperimetric inequality")
    evidence.append(
        ("perturbed simple loops", f"150/150; min defect={minimum_defect:.8g}")
    )

    for name, value in evidence:
        print(f"CHECK OK | {name}: {value}")
    print(
        "PROOF BOUNDARY: exact algebra checks the metric and physical factors; "
        "the global inequalities use standard Riemannian and isoperimetric theorems."
    )
    print("VERDICT: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - explicit audit output
        print(f"CHECK FAIL | {type(error).__name__}: {error}")
        traceback.print_exc()
        print("VERDICT: FAIL")
        sys.exit(1)
