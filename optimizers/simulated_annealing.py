"""Module for the Simulated Annealing algorithm."""
from typing import Callable, Tuple, List

import numpy as np


def simulated_annealing(
    objective_func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    n_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    n_dims: int,
    step_size: float,
    start_point: np.ndarray = None,
) -> Tuple[np.ndarray, float, List[float], List[np.ndarray], int]:
    """Simulated annealing algorithm.

    Args:
        objective_func (Callable[[np.ndarray], float]): The objective function to minimize.
        bounds (Tuple[float, float]): Tuple of lower and upper bounds for each dimension.
        n_iterations (int): The total number of iterations.
        initial_temp (float): The initial temperature.
        cooling_rate (float): The rate at which the temperature cools.
        n_dims (int): The number of dimensions of the problem.
        step_size (float): The size of the step to take for generating neighbors.
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
        current_solution = start_point
    else:
        current_solution = np.random.uniform(bounds[0], bounds[1], n_dims)

    current_temp = initial_temp
    # Инициализация: Начинаем с некоторого случайного решения и устанавливаем "температуру" на высокое значение.
    current_eval = objective_func(current_solution)
    n_evals += 1

    best_solution = current_solution
    best_eval = current_eval

    score_history = [best_eval]
    point_history = [best_solution]

    for i in range(n_iterations):
        # Generate a neighbor
        candidate = current_solution + np.random.normal(0, step_size, n_dims)
        # Делаем небольшой случайный шаг, чтобы получить новое "соседнее" решение.
        candidate = np.clip(candidate, bounds[0], bounds[1])
        candidate_eval = objective_func(candidate)
        n_evals += 1

        # Check if the new solution is better
        """
        Сравниваем его с текущим:
        Если новое решение лучше (энергия меньше), мы всегда его принимаем.
        Если новое решение хуже (энергия больше), мы все равно можем его принять! 
        Вероятность этого перехода зависит от того, насколько оно хуже и какая сейчас температура.
        """
        if candidate_eval < best_eval:
            best_solution, best_eval = candidate, candidate_eval
            current_solution, current_eval = candidate, candidate_eval
        else:
            # Calculate the metropolis acceptance criterion
            diff = candidate_eval - current_eval
            metropolis = np.exp(-diff / current_temp)
            if np.random.rand() < metropolis:
                current_solution, current_eval = candidate, candidate_eval
        """
        При высокой температуре алгоритм с большой вероятностью будет принимать даже плохие решения. Это позволяет ему "перепрыгивать" через холмы (локальные оптимумы) и не застревать в них, активно исследуя все пространство поиска.
        По мере снижения температуры вероятность принятия плохого решения уменьшается. Алгоритм становится более "жадным" и начинает фокусироваться на поиске самого лучшего решения в той области, где он находится.
        """
        # Cool the system
        current_temp *= cooling_rate
        score_history.append(best_eval)
        point_history.append(best_solution)

    return best_solution, best_eval, score_history, point_history, n_evals 