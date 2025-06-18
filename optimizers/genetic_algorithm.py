"""Module for the Genetic Algorithm."""
from typing import List, Callable, Tuple, Optional

import numpy as np


def selection(
    pop: List[np.ndarray], scores: List[float], k: int = 3
) -> np.ndarray:
    """Selects a parent from the population using tournament selection.

    Args:
        pop (List[np.ndarray]): The population of solutions.
        scores (List[float]): The scores of each solution in the population.
        k (int): The size of the tournament.

    Returns:
        np.ndarray: The selected parent.
    """
    # First random selection
    selection_ix = np.random.randint(len(pop))
    for _ in range(k - 1):
        # Second random selection
        ix = np.random.randint(len(pop))
        # Check if better (e.g. smaller score)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(
    p1: np.ndarray, p2: np.ndarray, r_cross: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs crossover between two parents.

    Args:
        p1 (np.ndarray): The first parent.
        p2 (np.ndarray): The second parent.
        r_cross (float): The crossover rate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the two children.
    """
    # Children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # Check for recombination
    if np.random.rand() < r_cross:
        # Проверяем, что длина вектора позволяет выполнить кроссовер
        if len(p1) > 2:
            # Select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(p1) - 1)
            # Perform crossover
            c1 = np.concatenate((p1[:pt], p2[pt:]))
            c2 = np.concatenate((p2[:pt], p1[pt:]))
        else:
            # Для малых размерностей (n_dims=2) просто обмениваем одно значение
            idx = np.random.randint(0, len(p1))
            c1[idx] = p2[idx]
            c2[idx] = p1[idx]
    return c1, c2


def mutation(bitstring: np.ndarray, r_mut: float, bounds: Tuple[float, float]) -> None:
    """Performs mutation on a bitstring.

    Args:
        bitstring (np.ndarray): The bitstring to mutate.
        r_mut (float): The mutation rate.
        bounds (Tuple[float, float]): The bounds for the values.
    """
    for i in range(len(bitstring)):
        # Check for a mutation
        if np.random.rand() < r_mut:
            # Replace with a new random value within bounds
            bitstring[i] = np.random.uniform(bounds[0], bounds[1])


def genetic_algorithm(
    objective: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    n_dims: int,
    n_iter: int,
    n_pop: int,
    r_cross: float,
    r_mut: float,
    initial_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, List[float], List[np.ndarray], int]:
    """Genetic algorithm for optimization.

    Args:
        objective (Callable[[np.ndarray], float]): The objective function to minimize.
        bounds (Tuple[float, float]): The lower and upper bounds for each dimension.
        n_dims (int): The number of dimensions.
        n_iter (int): The number of generations.
        n_pop (int): The size of the population.
        r_cross (float): The crossover rate.
        r_mut (float): The mutation rate.
        initial_point (Optional[np.ndarray]): Initial point to include in the population.

    Returns:
        Tuple[np.ndarray, float, List[float], List[np.ndarray], int]: A tuple
        containing the best solution found, its score, the history of scores,
        the history of points (best in each generation), and the number of
        function evaluations.
    """
    # Initialize function evaluation counter
    n_evals = 0
    # Initial population of random solutions
    if initial_point is not None:
        # Убедимся, что начальная точка имеет правильную размерность
        if len(initial_point) == n_dims:
            # Включаем начальную точку в популяцию и добавляем случайные решения для остальной части
            pop = [initial_point.copy()]
            # Генерируем остальные решения для популяции
            pop.extend([np.random.uniform(bounds[0], bounds[1], n_dims) for _ in range(n_pop - 1)])
        else:
            # Если размерность не соответствует, игнорируем начальную точку
            pop = [np.random.uniform(bounds[0], bounds[1], n_dims) for _ in range(n_pop)]
    else:
        # Генерируем полностью случайную популяцию
        pop = [np.random.uniform(bounds[0], bounds[1], n_dims) for _ in range(n_pop)]

    # Keep track of best solution
    scores = [objective(c) for c in pop]
    n_evals += len(pop)

    best_idx = np.argmin(scores)
    best_solution, best_eval = pop[best_idx], scores[best_idx]

    score_history = [best_eval]
    point_history = [best_solution]

    # Enumerate generations
    for _ in range(n_iter):
        # Check for new best solution
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_eval:
            best_solution, best_eval = pop[best_idx], scores[best_idx]

        # Select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # Create the next generation
        children = []
        for i in range(0, n_pop, 2):
            # Get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # Crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # Mutation
                mutation(c, r_mut, bounds)
                # Store for next generation
                children.append(c)
        # Replace population
        pop = children
        # Evaluate all candidates in the new population
        scores = [objective(c) for c in pop]
        n_evals += len(pop)
        
        score_history.append(best_eval)
        point_history.append(best_solution)

    # Final check for the best solution
    best_idx = np.argmin(scores)
    if scores[best_idx] < best_eval:
        best_solution, best_eval = pop[best_idx], scores[best_idx]

    return best_solution, best_eval, score_history, point_history, n_evals 