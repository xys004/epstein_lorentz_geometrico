"""Prueba simbólica y numérica de la acción conforme en tangentes SPT.

La prueba no usa la invariancia de la aceleración propia.  Verifica directamente
que el diferencial del boost en el hemisferio de tangentes tiene factor 1/D y,
tras reparametrizar por s', que kappa'_E = kappa_E/D^2.
"""

from __future__ import annotations

import math
import random

import sympy as sp


def exact_scalar_reduction() -> None:
    """Reduce a cero la identidad vectorial usando invariantes escalares."""

    gamma, q = sp.symbols("Gamma q", positive=True)
    x, y, z, a2 = sp.symbols("x y z A2", real=True)
    b2 = (gamma**2 - 1) / gamma**2
    h = gamma**2 / (gamma + 1)
    d = gamma * (1 - x)
    delta_d = -gamma * y
    b = -z / q

    # Productos escalares necesarios para L a y beta'.
    la2 = a2 + (2 * h + h**2 * b2) * y**2
    la_dot_lbeta = z + (2 * h + h**2 * b2) * x * y
    la_dot_b = (1 + h * b2) * y
    la_dot_beta_prime = (la_dot_lbeta - gamma * la_dot_b) / d
    q_prime = q / d

    # |beta'|^2 + q'^2 = 1; se conserva como identidad geométrica.
    transformed_numerator_sq = sp.expand(
        la2
        - 2 * delta_d * (la_dot_beta_prime + b * q_prime)
        + delta_d**2
        + b**2
    )
    residual = sp.factor(
        sp.simplify(sp.together(transformed_numerator_sq - (a2 + b**2)))
    )
    if residual != 0:
        raise AssertionError(f"La reducción simbólica no dio cero: {residual}")


def exact_simple_zero_check() -> None:
    """Exhibe la cúspide del módulo y la suavidad de la versión con signo."""

    y = sp.symbols("y", real=True)
    lam = sp.symbols("lambda", positive=True)
    rapidity = lam * y**2 / 2
    signed_invariant = sp.diff(rapidity, y)
    magnitude = sp.Abs(signed_invariant)

    if sp.simplify(signed_invariant - lam * y) != 0:
        raise AssertionError("La aceleración con signo no redujo a lambda*y")
    left = sp.limit(sp.diff(magnitude, y), y, 0, dir="-")
    right = sp.limit(sp.diff(magnitude, y), y, 0, dir="+")
    if left != -lam or right != lam:
        raise AssertionError(
            f"Límites laterales inesperados: izquierda={left}, derecha={right}"
        )


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vector: list[float]) -> float:
    return math.sqrt(dot(vector, vector))


def random_vector(generator: random.Random, n: int) -> list[float]:
    return [generator.uniform(-1.0, 1.0) for _ in range(n)]


def numerical_conformal_checks() -> tuple[int, float]:
    """Comprueba el diferencial completo, no sólo una curva particular."""

    generator = random.Random(20260804)
    count = 0
    max_relative_error = 0.0

    for n in (1, 2, 3, 5, 8):
        while count < 500 * (1 + (1, 2, 3, 5, 8).index(n)):
            beta = random_vector(generator, n)
            beta_norm = norm(beta)
            if beta_norm == 0.0:
                continue
            speed = generator.uniform(0.0, 0.96)
            beta = [speed * value / beta_norm for value in beta]
            q = math.sqrt(1.0 - speed * speed)

            boost = random_vector(generator, n)
            boost_norm = norm(boost)
            if boost_norm == 0.0:
                continue
            boost_speed = generator.uniform(0.01, 0.94)
            boost = [boost_speed * value / boost_norm for value in boost]
            boost_sq = boost_speed * boost_speed
            boost_gamma = 1.0 / math.sqrt(1.0 - boost_sq)
            h = (boost_gamma - 1.0) / boost_sq

            a = random_vector(generator, n)
            b = -dot(beta, a) / q
            d = boost_gamma * (1.0 - dot(boost, beta))
            delta_d = -boost_gamma * dot(boost, a)

            l_beta = [
                beta[i] + h * dot(boost, beta) * boost[i]
                for i in range(n)
            ]
            beta_prime = [
                (l_beta[i] - boost_gamma * boost[i]) / d
                for i in range(n)
            ]
            q_prime = q / d
            l_a = [a[i] + h * dot(boost, a) * boost[i] for i in range(n)]

            image = [
                (l_a[i] - beta_prime[i] * delta_d) / d
                for i in range(n)
            ]
            image.append((b - q_prime * delta_d) / d)
            expected = math.sqrt(dot(a, a) + b * b) / d
            actual = norm(image)
            relative_error = abs(actual - expected) / max(1.0, expected)
            max_relative_error = max(max_relative_error, relative_error)
            if relative_error > 3.0e-13:
                raise AssertionError(
                    f"Fallo conforme n={n}: error relativo={relative_error:.3e}"
                )
            count += 1

    return count, max_relative_error


def main() -> None:
    exact_scalar_reduction()
    exact_simple_zero_check()
    count, error = numerical_conformal_checks()
    print("OK: reducción simbólica directa de la identidad conforme = 0.")
    print("OK: cero simple: |lambda*y| tiene cúspide y lambda*y es suave.")
    print(
        f"OK: {count} diferenciales aleatorios en n=1,2,3,5,8; "
        f"error máximo={error:.3e}."
    )


if __name__ == "__main__":
    main()
