"""Module for the Hill Climbing algorithm."""
from typing import Callable, Tuple, List

import numpy as np


def hill_climbing(
    objective_func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    n_iterations: int,
    step_size: float,
    n_dims: int,
    start_point: np.ndarray = None,
) -> Tuple[np.ndarray, float, List[float], List[np.ndarray], int]:
    """Simple hill climbing algorithm.

    Args:
        objective_func (Callable[[np.ndarray], float]): The objective function to minimize.
        bounds (Tuple[float, float]): A tuple of lower and upper bounds for each dimension.
        n_iterations (int): The total number of iterations.
        step_size (float): The std deviation of the normal distribution for neighbors.
        n_dims (int): The number of dimensions of the problem.
        start_point (np.ndarray, optional): The starting point for the optimization.
            If None, a random point is generated. Defaults to None.

    Returns:
        Tuple[np.ndarray, float, List[float], List[np.ndarray], int]: A tuple
        containing the best solution found, its score, the history of scores,
        the history of points, and the number of function evaluations.
    """
    # Initialize function evaluation counter
    n_evals = 0

    # Generate an initial point or use the provided one
    if start_point is not None:
        solution = start_point
    else:
        solution = np.random.uniform(bounds[0], bounds[1], n_dims)

    solution_eval = objective_func(solution)
    n_evals += 1

    # Run the hill climb
    score_history = [solution_eval]
    point_history = [solution]
    for _ in range(n_iterations):
        # Take a step
        candidate = solution + np.random.normal(0, step_size, n_dims)
        # Clip to bounds
        candidate = np.clip(candidate, bounds[0], bounds[1])
        # Evaluate candidate point
        candidate_eval = objective_func(candidate)
        n_evals += 1
        # Check if we should keep the new point
        if candidate_eval < solution_eval:
            # Store the new point
            solution, solution_eval = candidate, candidate_eval

        score_history.append(solution_eval)
        point_history.append(solution)

    return solution, solution_eval, score_history, point_history, n_evals 