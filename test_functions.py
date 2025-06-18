from typing import Dict, Callable, Tuple

import numpy as np


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0))


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    sum_sq_term = -0.2 * np.sqrt(np.sum(x**2) / n)
    cos_term = np.sum(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(sum_sq_term) - np.exp(cos_term / n) + 20.0 + np.exp(1)
    )


def booth(x: np.ndarray) -> float:
    if len(x) != 2:
        raise ValueError("Booth function is only defined for 2 dimensions.")
    return float((x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2)


def booth_noisy(x: np.ndarray, noise_level: float = 1.0) -> float:
    base_value = booth(x)
    noise = np.random.normal(0, noise_level)
    return base_value + noise



FUNCTIONS: Dict[str, Dict[str, Callable | Tuple[float, float]]] = {
    "sphere": {"func": sphere, "bounds": (-5.12, 5.12)},
    "rosenbrock": {"func": rosenbrock, "bounds": (-2.048, 2.048)},
    "rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12)},
    "ackley": {"func": ackley, "bounds": (-32.768, 32.768)},
    "booth": {"func": booth, "bounds": (-10, 10)},
    "booth_noisy": {"func": booth_noisy, "bounds": (-10, 10)},
} 