import os
import pickle
import numpy as np

import pandas as pd

from plots import plot_contour_and_path
from test_functions import FUNCTIONS


def get_zoomed_bounds(func_name: str, history_points: list) -> tuple:
    """
    вычисляет границы для приближения пути.
    
    Args:
        func_name (str): название функции, которая будет изображена
        history_points (list): список точек в оптимизационном пути
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max) границы для изображения
    """
    # преобразуем историю в numpy массив
    points = np.array(history_points)
    
    # находим минимальное и максимальное значения по каждой оси
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # находим диапазон значений по каждой оси
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    # находим центр пути
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # для более сильного приближения используем меньший отступ
    # минимальный размер окна просмотра для случая, если путь очень короткий
    min_range = 0.5  
    
    # добавляем отступ в зависимости от диапазона
    if range_x < min_range:
        # для очень короткого пути используем фиксированный размер окна просмотра
        min_x = center_x - min_range
        max_x = center_x + min_range
    else:
        # для более длинного пути добавляем отступ в 10% от размера (вместо 20%)
        pad_x = range_x * 0.1
        min_x = min_x - pad_x
        max_x = max_x + pad_x
        
    if range_y < min_range:
        min_y = center_y - min_range
        max_y = center_y + min_range
    else:
        pad_y = range_y * 0.1
        min_y = min_y - pad_y
        max_y = max_y + pad_y
    
    # специальные случаи для разных функций
    
    # для функции Booth, если мы близко к глобальному минимуму (1, 3)
    # создаем более плотное приближение
    if func_name == 'booth' and abs(center_x - 1) < 2 and abs(center_y - 3) < 2:
        # если мы близко к оптимуму - сильно приближаем график
        zoom_level = 0.8  # меньшее значение = более сильное приближение
        if range_x < 0.1 and range_y < 0.1:
            # если путь очень короткий, центрируем вид на глобальном минимуме
            return (0.7, 1.3, 2.7, 3.3)
        else:
            # иначе центрируемся на пути с небольшим отступом
            return (min_x - range_x * zoom_level, max_x + range_x * zoom_level,
                    min_y - range_y * zoom_level, max_y + range_y * zoom_level)
    
    # для функции Ackley, которая имеет много локальных минимумов:
    # - если мы близко к глобальному минимуму (0, 0), сильнее приближаем
    # - если путь длинный и проходит через разные области, увеличиваем масштаб
    if func_name == 'ackley':
        if abs(center_x) < 1 and abs(center_y) < 1:
            # близко к глобальному минимуму - хорошее приближение
            return (center_x - 1, center_x + 1, center_y - 1, center_y + 1)
        elif range_x < 5 and range_y < 5:
            # умеренный путь - показываем немного шире для контекста
            return (center_x - range_x * 0.6, center_x + range_x * 0.6,
                    center_y - range_y * 0.6, center_y + range_y * 0.6)
        else:
            # для дальних точек, особенно (-15, 15), ограничиваем максимальный диапазон
            if abs(min_x) > 10 or abs(min_y) > 10 or abs(max_x) > 10 or abs(max_y) > 10:
                # масштабируем относительно центра пути, а не абсолютных координат
                # чтобы избежать слишком большого масштабирования
                return (center_x - 5, center_x + 5, center_y - 5, center_y + 5)
            else:
                # длинный путь, но не хотим показывать слишком много
                return (min_x - range_x * 0.2, max_x + range_x * 0.2,
                        min_y - range_y * 0.2, max_y + range_y * 0.2)
    
    # для зашумленной функции Booth нам нужно немного больше контекста
    if func_name == 'booth_noisy':
        # немного больший масштаб для понимания шумов
        return (min_x - range_x * 0.15, max_x + range_x * 0.15,
                min_y - range_y * 0.15, max_y + range_y * 0.15)

    return (min_x, max_x, min_y, max_y)


def create_coords_string(point):
    """создает строку с координатами для включения в имя файла"""
    if point is None:
        return "random"
    
    # для многомерных векторов возвращаем только первые 2-3 координаты
    if len(point) > 3:
        coords = point[:3]
        return f"x{coords[0]:.1f}_y{coords[1]:.1f}_z{coords[2]:.1f}"
    elif len(point) == 3:
        return f"x{point[0]:.1f}_y{point[1]:.1f}_z{point[2]:.1f}"
    else:
        return f"x{point[0]:.1f}_y{point[1]:.1f}"


def create_special_view_for_booth_near_optimum(path_history, title, save_path):
    """
    создает дополнительный приближенный график для функции Booth, 
    очень близко к глобальному минимуму для визуализации финальной сходимости.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from test_functions import booth
    
    # создаем отдельный очень приближенный вид 
    plt.figure(figsize=(10, 8))
    
    # определяем границы (очень близко к глобальному минимуму в точке (1, 3))
    x_min, x_max = 0.9, 1.1
    y_min, y_max = 2.9, 3.1
    
    # создаем сетку для контурного графика
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # вычисляем значения функции на сетке
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = booth(np.array([X[i, j], Y[i, j]]))
    
    # рисуем контурный график
    cp = plt.contour(X, Y, Z, 20, cmap='viridis')
    plt.clabel(cp, inline=1, fontsize=8)
    plt.colorbar(label='Function Value')
    
    # фильтруем точки пути, которые попадают в заданный диапазон
    points = np.array(path_history)
    mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
    
    filtered_path = points[mask]
    
    if len(filtered_path) > 0:
        # рисуем путь
        plt.plot(filtered_path[:, 0], filtered_path[:, 1], 'r-o', 
                markersize=4, linewidth=1, label='Path')
        
        # отмечаем первую и последнюю точку отфильтрованного пути
        plt.plot(filtered_path[0, 0], filtered_path[0, 1], 'go', 
                markersize=6, label='First in view')
        plt.plot(filtered_path[-1, 0], filtered_path[-1, 1], 'bo', 
                markersize=6, label='Last point')
    
    # отмечаем глобальный минимум
    plt.plot(1, 3, 'r*', markersize=10, label='Global Minimum (1,3)')
    
    plt.title(title + '\n(Очень близкое приближение к глобальному минимуму)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.legend()
    
    # сохраняем с суффиксом ultra_zoom
    ultra_zoom_path = save_path.replace('.png', '_ultra_zoom.png')
    plt.savefig(ultra_zoom_path)
    plt.close()
    print(f"График сохранен: {ultra_zoom_path}")


results_dir = "results"
results_file = os.path.join(results_dir, "detailed_results.pkl")

if not os.path.exists(results_file):
    print(f"Файл с результатами не найден по пути {results_file}.")
    print("Пожалуйста, запустите run_experiments.py сначала.")
    exit()

with open(results_file, "rb") as f:
    results_data = pickle.load(f)

df = pd.DataFrame(results_data)

# мы хотим изобразить только 2D функции, так как они легко визуализируются
funcs_to_plot = df[df["n_dims"] == 2]["function"].unique()

for func_name in funcs_to_plot:
    print(f"--- Изображение для функции: {func_name} ---")
    func_info = FUNCTIONS[func_name]
    objective_func = func_info["func"]
    bounds = func_info["bounds"]

    # создаем поддиректорию для функции
    func_dir = os.path.join(results_dir, func_name)
    os.makedirs(func_dir, exist_ok=True)

    # фильтруем данные для текущей функции
    func_df = df[df["function"] == func_name]

    for _, row in func_df.iterrows():
        algo_name = row["algorithm"]
        start_point_name = row["start_point_name"]
        path_history = row["path_history"]
        score = row["score"]
        
        # получаем первую точку пути (начальная точка)
        first_point = path_history[0] if path_history else None
        
        # создаем строку с координатами для имени файла
        coords_str = create_coords_string(first_point)
        
        title = (
            f"{algo_name} на {func_name.capitalize()}\n"
            f"Старт: {start_point_name} ({coords_str}) | Финальное значение: {score:.4f}"
        )
        
        # сохраняем в поддиректорию для функции с указанием координат в имени файла
        save_path = os.path.join(
            func_dir, f"{algo_name}_{start_point_name}_{coords_str}.png"
        )

        print(f"  Изображение для {algo_name} из {start_point_name} ({coords_str})...")
        
        # получаем приближенные границы для отображения
        zoomed_x_min, zoomed_x_max, zoomed_y_min, zoomed_y_max = get_zoomed_bounds(func_name, path_history)

        # сначала сохраняем приближенный график
        plot_contour_and_path(
            func=lambda x, y, **kwargs: objective_func(np.array([x, y])),
            params={},
            history=path_history,
            title=title,
            x_lim=(zoomed_x_min, zoomed_x_max),
            y_lim=(zoomed_y_min, zoomed_y_max),
            save_path=save_path,
            show_plot=False,
        )
        
        # затем сохраняем полный график для сравнения
        full_view_path = os.path.join(
            func_dir, f"{algo_name}_{start_point_name}_{coords_str}_full_view.png"
        )
        plot_contour_and_path(
            func=lambda x, y, **kwargs: objective_func(np.array([x, y])),
            params={},
            history=path_history,
            title=title + "\n(Полный вид)",
            x_lim=bounds,
            y_lim=bounds,
            save_path=full_view_path,
            show_plot=False,
        )
        
        # для функции Booth, если найденное значение близко к минимуму,
        # создаем специальный ультра-приближенный график, чтобы показать финальную сходимость
        # применяем не только для Hill Climbing, но и для всех алгоритмов
        if func_name == "booth" and score < 1.0:
            create_special_view_for_booth_near_optimum(path_history, title, save_path)
        
print("\nВсе графики сгенерированы и сохранены в поддиректориях директории 'results'.")
