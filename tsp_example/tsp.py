"""Solves the TSP using Simulated Annealing."""
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def calculate_total_distance(tour: List[int], cities: np.ndarray) -> float:
    """Calculates the total distance of a tour.

    Args:
        tour (List[int]): A list of city indices in order.
        cities (np.ndarray): An array of city coordinates.

    Returns:
        float: The total distance of the tour.
    """
    total_distance = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        city1_idx = tour[i]
        city2_idx = tour[(i + 1) % num_cities]
        total_distance += float(np.linalg.norm(cities[city1_idx] - cities[city2_idx]))
    return total_distance


def swap_2_opt(tour: List[int]) -> List[int]:
    """Generates a neighbor tour by swapping two cities.

    Args:
        tour (List[int]): The current tour.

    Returns:
        List[int]: A new tour with two cities swapped.
    """
    new_tour = tour[:]
    i, j = random.sample(range(len(new_tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def solve_tsp_with_sa(
    cities: np.ndarray, n_iterations: int, initial_temp: float, cooling_rate: float
) -> Tuple[List[int], float, List[float]]:
    """Solves the TSP using Simulated Annealing.

    Args:
        cities (np.ndarray): An array of city coordinates.
        n_iterations (int): The number of iterations to run.
        initial_temp (float): The initial temperature.
        cooling_rate (float): The cooling rate.

    Returns:
        Tuple[List[int], float, List[float]]: The best tour found, its distance,
        and the history of distances.
    """
    # Initial solution
    current_tour = list(range(len(cities)))
    random.shuffle(current_tour)
    current_distance = calculate_total_distance(current_tour, cities)

    best_tour = current_tour[:]
    best_distance = current_distance

    temp = initial_temp
    history = [current_distance]

    for _ in range(n_iterations):
        # Generate a neighbor
        candidate_tour = swap_2_opt(current_tour)
        candidate_distance = calculate_total_distance(candidate_tour, cities)

        # Metropolis acceptance criterion
        acceptance_prob = np.exp((current_distance - candidate_distance) / temp)
        if candidate_distance < current_distance or random.random() < acceptance_prob:
            current_tour = candidate_tour[:]
            current_distance = candidate_distance

        if current_distance < best_distance:
            best_tour = current_tour[:]
            best_distance = current_distance

        temp *= cooling_rate
        history.append(best_distance)

    return best_tour, best_distance, history


def plot_tour(tour: List[int], cities: np.ndarray, title: str = "TSP Solution") -> None:
    """Plots the cities and the optimized tour.

    Args:
        tour (List[int]): The tour to plot.
        cities (np.ndarray): An array of city coordinates.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 8))

    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c="red", zorder=2)
    for i, city in enumerate(cities):
        plt.text(city[0] + 0.1, city[1] + 0.1, str(i))

    # Plot tour
    for i in range(len(tour)):
        start_node = tour[i]
        end_node = tour[(i + 1) % len(tour)]
        plt.plot(
            [cities[start_node, 0], cities[end_node, 0]],
            [cities[start_node, 1], cities[end_node, 1]],
            "b-",
        )

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig("results/tsp_solution.png")
    plt.show()


def main() -> None:
    """Main function to run the TSP solver."""
    # --- Parameters ---
    num_cities = 25
    n_iterations_tsp = 15000
    initial_temp_tsp = 100.0
    cooling_rate_tsp = 0.999

    # Generate random cities
    cities = np.random.rand(num_cities, 2) * 10

    print("Solving TSP with Simulated Annealing...")
    best_tour, best_distance, history = solve_tsp_with_sa(
        cities, n_iterations_tsp, initial_temp_tsp, cooling_rate_tsp
    )

    print(f"Best tour distance: {best_distance:.4f}")

    # Plotting
    plot_tour(best_tour, cities, f"TSP Solution (Distance: {best_distance:.2f})")

    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("TSP Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.savefig("results/tsp_convergence.png")
    plt.show()


if __name__ == "__main__":
    main() 