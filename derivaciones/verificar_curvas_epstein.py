"""Verificaciones de la geometría diferencial de curvas de Epstein.

Usa solo la biblioteca estándar. Todas las cantidades son adimensionales con c=1,
salvo los nombres que recuerdan su interpretación física.
"""

from __future__ import annotations

import math
import random


TOL = 2.0e-12


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vector: list[float]) -> float:
    return math.sqrt(dot(vector, vector))


def assert_close(actual: float, expected: float, label: str) -> None:
    scale = max(1.0, abs(actual), abs(expected))
    error = abs(actual - expected)
    if error > TOL * scale:
        raise AssertionError(
            f"{label}: actual={actual:.16g}, esperado={expected:.16g}, "
            f"error={error:.3e}"
        )


def check_catenary() -> None:
    """Aceleración propia constante produce una catenaria SPT."""

    radius = 2.7
    for arc_length in (-7.0, -2.0, 0.0, 1.5, 8.0):
        root = math.sqrt(radius * radius + arc_length * arc_length)
        x_s = arc_length / root
        y_s = radius / root
        curvature = radius / (radius * radius + arc_length * arc_length)

        assert_close(x_s * x_s + y_s * y_s, 1.0, "velocidad unitaria")
        assert_close(
            curvature / (y_s * y_s),
            1.0 / radius,
            "curvatura ponderada constante",
        )

        y = radius * math.asinh(arc_length / radius)
        x = root - radius
        assert_close(
            x,
            radius * (math.cosh(y / radius) - 1.0),
            "ecuación cartesiana de la catenaria",
        )
        assert_close(
            math.atan2(x_s, y_s),
            math.atan(math.sinh(y / radius)),
            "gudermanniano como ángulo tangente",
        )


def check_circle_false_friend() -> None:
    """Curvatura euclídea constante no implica aceleración propia constante."""

    radius = 3.4
    for theta in (-1.1, -0.4, 0.0, 0.65, 1.2):
        q = math.cos(theta)
        beta = math.sin(theta)
        curvature = 1.0 / radius
        proper_acceleration = curvature / (q * q)

        assert q > 0.0
        assert_close(beta * beta + q * q, 1.0, "círculo unitario")
        assert_close(
            proper_acceleration,
            1.0 / (radius * math.cos(theta) ** 2),
            "aceleración propia del círculo SPT",
        )


def check_uniform_circular_motion() -> None:
    """El movimiento circular uniforme se representa por una hélice SPT."""

    radius = 2.3
    beta = 0.72
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    angular_frequency = beta / radius

    euclidean_curvature = beta * beta / radius
    euclidean_torsion = angular_frequency / gamma
    proper_acceleration = euclidean_curvature * gamma * gamma

    assert_close(
        proper_acceleration,
        gamma * gamma * beta * beta / radius,
        "aceleración propia centrípeta",
    )

    # Fórmulas estándar de una hélice de radio R y paso h por radián.
    pitch = 1.0 / (angular_frequency * gamma)
    helix_curvature = radius / (radius * radius + pitch * pitch)
    helix_torsion = pitch / (radius * radius + pitch * pitch)
    assert_close(
        helix_curvature,
        euclidean_curvature,
        "curvatura de la hélice SPT",
    )
    assert_close(
        helix_torsion,
        euclidean_torsion,
        "torsión de la hélice SPT",
    )


def check_boosted_catenary() -> None:
    """Un boost desplaza la catenaria, pero preserva kappa/q^2."""

    radius = 1.9
    for boost_rapidity in (-0.9, -0.25, 0.55, 1.0):
        for proper_rapidity in (-1.4, -0.2, 0.0, 0.8, 1.6):
            shifted = proper_rapidity - boost_rapidity
            q_prime = 1.0 / math.cosh(shifted)
            curvature_prime = 1.0 / (
                radius * math.cosh(shifted) ** 2
            )
            assert_close(
                curvature_prime / (q_prime * q_prime),
                1.0 / radius,
                "invariante de la catenaria bajo boost",
            )

            s_prime = radius * (
                math.sinh(shifted) + math.sinh(boost_rapidity)
            )
            x_prime = radius * (
                math.cosh(shifted) - math.cosh(boost_rapidity)
            )
            y = radius * proper_rapidity
            assert_close(
                x_prime + radius * math.cosh(boost_rapidity),
                radius
                * math.cosh(y / radius - boost_rapidity),
                "ecuación cartesiana de la catenaria boosted",
            )
            assert_close(
                (
                    s_prime
                    - radius * math.sinh(boost_rapidity)
                )
                / radius,
                math.sinh(shifted),
                "parametrización temporal boosted",
            )


def check_curvature_dictionary_3d(
    beta: list[float], beta_s: list[float]
) -> tuple[float, float]:
    """Comprueba a_prop/c^2 = gamma^2*kappa_E en tres dimensiones."""

    beta_sq = dot(beta, beta)
    if not 0.0 <= beta_sq < 1.0:
        raise ValueError("La velocidad debe ser sublumínica")

    q = math.sqrt(1.0 - beta_sq)
    gamma = 1.0 / q
    q_s = -dot(beta, beta_s) / q
    curvature = math.sqrt(dot(beta_s, beta_s) + q_s * q_s)

    # Con c=1, a_coord = d beta / dt = d beta / ds = beta_s.
    coordinate_acceleration = beta_s
    proper_acceleration = gamma * gamma * math.sqrt(
        dot(coordinate_acceleration, coordinate_acceleration)
        + gamma
        * gamma
        * dot(beta, coordinate_acceleration)
        * dot(beta, coordinate_acceleration)
    )

    assert_close(
        proper_acceleration,
        gamma * gamma * curvature,
        "diccionario curvatura-aceleración 3+1",
    )
    assert_close(
        proper_acceleration,
        curvature / (q * q),
        "curvatura ponderada 3+1",
    )
    return curvature, q


def check_general_boost(
    beta: list[float], beta_s: list[float], boost: list[float]
) -> None:
    """Verifica la ley de curvatura bajo un boost vectorial general."""

    curvature, q = check_curvature_dictionary_3d(beta, beta_s)
    boost_sq = dot(boost, boost)
    if not 0.0 < boost_sq < 1.0:
        raise ValueError("El boost debe ser sublumínico y no nulo")

    boost_gamma = 1.0 / math.sqrt(1.0 - boost_sq)
    boost_dot_beta = dot(boost, beta)
    boost_dot_beta_s = dot(boost, beta_s)

    parameter_jacobian = boost_gamma * (1.0 - boost_dot_beta)
    parameter_jacobian_s = -boost_gamma * boost_dot_beta_s

    tangent_spatial_numerator = [
        beta[index]
        + (
            (boost_gamma - 1.0) * boost_dot_beta / boost_sq
            - boost_gamma
        )
        * boost[index]
        for index in range(3)
    ]
    tangent_spatial_numerator_s = [
        beta_s[index]
        + (boost_gamma - 1.0)
        * boost_dot_beta_s
        / boost_sq
        * boost[index]
        for index in range(3)
    ]

    beta_prime = [
        component / parameter_jacobian
        for component in tangent_spatial_numerator
    ]
    q_prime = q / parameter_jacobian

    beta_prime_s = [
        (
            tangent_spatial_numerator_s[index] * parameter_jacobian
            - tangent_spatial_numerator[index] * parameter_jacobian_s
        )
        / (parameter_jacobian * parameter_jacobian)
        for index in range(3)
    ]
    beta_prime_s_prime = [
        component / parameter_jacobian for component in beta_prime_s
    ]

    q_s = -dot(beta, beta_s) / q
    q_prime_s = (
        q_s * parameter_jacobian - q * parameter_jacobian_s
    ) / (parameter_jacobian * parameter_jacobian)
    q_prime_s_prime = q_prime_s / parameter_jacobian

    curvature_prime = math.sqrt(
        dot(beta_prime_s_prime, beta_prime_s_prime)
        + q_prime_s_prime * q_prime_s_prime
    )

    assert_close(
        dot(beta_prime, beta_prime) + q_prime * q_prime,
        1.0,
        "el boost preserva curvas unitarias",
    )
    assert_close(
        curvature_prime,
        curvature / (parameter_jacobian * parameter_jacobian),
        "ley de transformación de la curvatura",
    )
    assert_close(
        curvature_prime / (q_prime * q_prime),
        curvature / (q * q),
        "invariancia de la curvatura ponderada",
    )


def run_deterministic_random_checks() -> None:
    generator = random.Random(20260726)
    checked = 0
    while checked < 250:
        beta = [generator.uniform(-0.65, 0.65) for _ in range(3)]
        boost = [generator.uniform(-0.55, 0.55) for _ in range(3)]
        if dot(beta, beta) >= 0.88 or not 1.0e-8 < dot(boost, boost) < 0.82:
            continue

        beta_s = [generator.uniform(-0.4, 0.4) for _ in range(3)]
        check_general_boost(beta, beta_s, boost)
        checked += 1


def main() -> None:
    check_catenary()
    check_circle_false_friend()
    check_uniform_circular_motion()
    check_boosted_catenary()
    run_deterministic_random_checks()
    print("OK: catenaria SPT verificada.")
    print("OK: círculo SPT y su aceleración propia variable verificados.")
    print("OK: hélice SPT del movimiento circular uniforme verificada.")
    print("OK: familia boosted de catenarias verificada.")
    print("OK: a_prop/c^2 = kappa_E/q^2 verificado en 3+1.")
    print("OK: acción de boosts y ley kappa' = kappa/(ds'/ds)^2 verificadas.")
    print("OK: 250 casos vectoriales deterministas pasaron.")


if __name__ == "__main__":
    main()
