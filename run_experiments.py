
import os
import pickle
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from optimizers.genetic_algorithm import genetic_algorithm
from optimizers.hill_climbing import hill_climbing
from optimizers.simulated_annealing import simulated_annealing
from test_functions import FUNCTIONS

N_ITERATIONS = 500  # уменьшено для более быстрых тестов
N_RUNS = 3  # количество различных наборов стартовых точек
RESULTS_DIR = "results"

STARTING_POINT_SETS = {
    "ackley": [
        {"name": "Center", "point": np.array([0.0, 0.0])},
        {"name": "Near Global Minimum", "point": np.array([0.5, -0.5])},
        {"name": "Far Point", "point": np.array([-15.0, 15.0])},
    ],
    
    # для Booth функции (глобальный минимум в (1, 3))
    "booth": [
        {"name": "Center", "point": np.array([0.0, 0.0])},
        {"name": "Near Optimum", "point": np.array([1.2, 3.2])},
        {"name": "Far Point", "point": np.array([-8.0, 8.0])},
    ],
    
    # для зашумленной Booth функции
    "booth_noisy": [
        {"name": "Center", "point": np.array([0.0, 0.0])},
        {"name": "Near Optimum", "point": np.array([1.2, 3.2])},
        {"name": "Far Point", "point": np.array([-8.0, 8.0])},
    ],
    
    # для многомерных функций используем случайные точки
    "sphere": None,
    "rastrigin": None,
}

RunResult = Dict[str, Any]
FullResults = List[RunResult]


def run_experiments() -> FullResults:
    """запускает полный набор экспериментов по оптимизации"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    results_data: FullResults = []

    # фильтр для функций, которые мы хотим протестировать
    functions_to_test = {
        "ackley": 2,
        "booth": 2,
        "booth_noisy": 2,
        "sphere": 5,  # тестирование с 5 измерениями
        "rastrigin": 5,
    }

    for func_name, n_dims in tqdm(functions_to_test.items(), desc="Functions"):
        func_data = FUNCTIONS[func_name]
        objective = func_data["func"]
        bounds = func_data["bounds"]
        
        # определяем стартовые точки
        start_points_info = []
        
        if n_dims == 2 and STARTING_POINT_SETS.get(func_name):
            # для 2D функций используем предопределенные начальные точки
            start_points_info = STARTING_POINT_SETS[func_name]
        else:
            # для многомерных функций генерируем случайные начальные точки
            for i in range(N_RUNS):
                random_point = np.random.uniform(bounds[0], bounds[1], n_dims)
                start_points_info.append({
                    "name": f"Random Set {i+1}",
                    "point": random_point
                })
        
        # для каждого набора начальных точек запускаем все алгоритмы
        for point_info in start_points_info:
            start_point_name = point_info["name"]
            start_point = point_info["point"]
            
            # --- Hill Climbing ---
            print(f"  Running Hill Climbing from {start_point_name}...")
            _, score, score_hist, path, evals = hill_climbing(
                objective, bounds, N_ITERATIONS, 0.2, n_dims, start_point=start_point
            )
            results_data.append({
                "function": func_name, "algorithm": "Hill Climbing", "n_dims": n_dims,
                "start_point_name": start_point_name, "score": score,
                "n_iterations": N_ITERATIONS, "n_evals": evals,
                "score_history": score_hist, "path_history": path
            })

            # --- Simulated Annealing ---
            print(f"  Running Simulated Annealing from {start_point_name}...")
            _, score, score_hist, path, evals = simulated_annealing(
                objective, bounds, N_ITERATIONS, 1000, 0.99, n_dims, 0.2, start_point=start_point
            )
            results_data.append({
                "function": func_name, "algorithm": "Simulated Annealing", "n_dims": n_dims,
                "start_point_name": start_point_name, "score": score,
                "n_iterations": N_ITERATIONS, "n_evals": evals,
                "score_history": score_hist, "path_history": path
            })
            
            # --- Genetic Algorithm ---
            # будем запускать для каждой начальной точки, используя её как одно из решений в популяции
            print(f"  Running Genetic Algorithm from {start_point_name}...")
            ga_generations = N_ITERATIONS // 50
            _, score, score_hist, path, evals = genetic_algorithm(
                objective, bounds, n_dims, ga_generations, 50, 0.9, 0.1, 
                initial_point=start_point  # передаем начальную точку
            )
            results_data.append({
                "function": func_name, "algorithm": "Genetic Algorithm", "n_dims": n_dims,
                "start_point_name": start_point_name, "score": score,
                "n_iterations": ga_generations, "n_evals": evals,
                "score_history": score_hist, "path_history": path
            })
            
    # сохраняем подробные результаты для изображения
    with open(os.path.join(RESULTS_DIR, "detailed_results.pkl"), "wb") as f:
        pickle.dump(results_data, f)

    return results_data


def save_summary_to_csv(results_data: FullResults) -> None:
    """сохраняет сводку результатов в CSV файл.

    Args:
        results_data (FullResults): результаты всех оптимизационных запусков.
    """
    # мы не нужно историю для таблицы сводки
    summary_list = []
    for r in results_data:
        summary_list.append({k: v for k, v in r.items() if k not in ["score_history", "path_history"]})

    df = pd.DataFrame(summary_list)
    
    # группируем по функции, алгоритму, начальной точке и вычисляем статистику
    grouped = df.groupby(["function", "algorithm", "start_point_name", "n_dims"])
    
    agg_df = grouped.agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        best_score=("score", "min"),
        mean_iterations=("n_iterations", "mean"),
        mean_evals=("n_evals", "mean"),
    ).reset_index()

    csv_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    agg_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print(" " * 15 + "PERFORMANCE SUMMARY")
    print("=" * 60)
    print(agg_df.to_string())
    print("=" * 60)
    print(f"\nТаблица сводки сохранена в '{csv_path}'")


results = run_experiments()
save_summary_to_csv(results)
