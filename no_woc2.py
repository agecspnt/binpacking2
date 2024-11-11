import random
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
from items import get_test_data
import csv
import os
from PIL import Image
import shutil
import json

def save_population_agreement_matrix(population, num_items, filename='population_agreement_matrix.txt'):
    agreement_matrix = np.zeros((num_items, num_items))

    for solution in population:
        for i in range(len(solution) - 1):
            item1, item2 = solution[i], solution[i + 1]
            agreement_matrix[item1][item2] += 1
            agreement_matrix[item2][item1] += 1

    matrix_str = "Population Agreement Matrix (Final Generation):\n"
    matrix_str += f"Population Size: {len(population)}\n"
    matrix_str += "-" * (num_items * 8) + "\n"

    matrix_str += "     "
    for i in range(num_items):
        matrix_str += f"{i:6d} "
    matrix_str += "\n"
    matrix_str += "-" * (num_items * 8) + "\n"

    for i in range(num_items):
        matrix_str += f"{i:3d} |"
        for j in range(num_items):
            matrix_str += f"{agreement_matrix[i][j]:6.3f} "
        matrix_str += "\n"

    with open(filename, 'w') as f:
        f.write(matrix_str)

    return agreement_matrix

class GA:
    def __init__(self, num_items, num_total, max_iterations, items, mutation_rate=0.05, crossover_rate=0.85,
                 population_size=25, early_stop_generations=200, improvement_threshold=0.001, bin_width=10,
                 bin_height=10, update_callback=None, selection_method='roulette', mutation_type='reverse',
                 crossover_type='order', save_outputs=False, run_number=1, output_folder='experiment_results'):
        # 添加run_number和output_folder参数
        self.num_items = num_items
        self.num_total = num_total
        self.max_iterations = max_iterations
        self.items = items
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = mutation_rate
        self.crossover_rate = crossover_rate
        self.clone_rate = 1 - crossover_rate - mutation_rate
        self.population_size = population_size
        self.early_stop_generations = early_stop_generations
        self.improvement_threshold = improvement_threshold
        self.best_fitness = float('inf')
        self.best_solution = None
        self.improvement_curve = []
        self.fruits = self.init_population()
        self.update_callback = update_callback
        self.selection_method = selection_method
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.generation = 0
        self.save_outputs = save_outputs
        self.run_number = run_number
        self.output_folder = output_folder

        # Create run-specific directory for images
        if self.save_outputs:
            self.run_output_dir = os.path.join(output_folder, f'run_{run_number}')
            self.images_dir = os.path.join(self.run_output_dir, 'generation_images')
            if os.path.exists(self.images_dir):
                shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir, exist_ok=True)

    def save_generation_image(self, bins, improvement_curve):
        if not self.save_outputs:
            return

        fig = plt.figure(figsize=(8, 10))
        ax2 = fig.add_axes([0.1, 0.35, 0.8, 0.6])
        ax1 = fig.add_axes([0.1, 0.05, 0.8, 0.25])

        # Plot improvement curve
        ax1.plot(improvement_curve)
        ax1.set_title('Fitness Improvement')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.grid(True)

        # Plot bin packing solution
        colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, 300))
        bins_per_row = 4
        spacing = 2
        grid_width = bins_per_row * (self.bin_width + spacing)
        num_rows = math.ceil(len(bins) / bins_per_row)

        for i, bin_dict in enumerate(bins):
            row = i // bins_per_row
            col = i % bins_per_row

            x_pos = col * (self.bin_width + spacing)
            y_pos = (num_rows - 1 - row) * (self.bin_height + spacing)

            bin_rect = plt.Rectangle((x_pos, y_pos), self.bin_width, self.bin_height,
                                   facecolor='none', edgecolor='black', linestyle='--')
            ax2.add_patch(bin_rect)

            for item in bin_dict['items']:
                item_x = x_pos + item['x']
                item_y = y_pos + item['y']
                item_w = item['width']
                item_h = item['height']
                color = colors[item['id'] - 1]

                rect = plt.Rectangle((item_x, item_y), item_w, item_h,
                                   facecolor=color, edgecolor='black', alpha=0.7)
                ax2.add_patch(rect)
                ax2.text(item_x + item_w / 2, item_y + item_h / 2, f'{item["id"]}',
                         ha='center', va='center')

        ax2.set_xlim(-spacing, grid_width + spacing)
        ax2.set_ylim(-spacing, num_rows * (self.bin_height + spacing) + spacing)
        ax2.set_title(f'Current Best Solution (Generation {self.generation})')
        ax2.set_aspect('equal')
        ax2.grid(True)

        plt.savefig(os.path.join(self.images_dir, f'generation_{self.generation:04d}.png'))
        plt.close(fig)

    def create_gif(self):
        if not self.save_outputs or not os.path.exists(self.images_dir):
            return

        images = []
        image_files = sorted(os.listdir(self.images_dir))

        for filename in image_files:
            if filename.endswith('.png'):
                file_path = os.path.join(self.images_dir, filename)
                images.append(Image.open(file_path))

        if images:
            gif_path = os.path.join(self.run_output_dir, f'evolution_run_{self.run_number}.gif')
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0
            )

    def init_population(self):
        population = []
        for _ in range(self.num_total):
            solution = list(range(self.num_items))
            random.shuffle(solution)
            population.append(solution)
        return population

    def pack_items(self, solution):
        bins = []
        current_bin = {'items': [], 'spaces': [(0, 0, self.bin_width, self.bin_height)]}

        for item_idx in solution:
            item = self.items[item_idx]
            if not self.try_pack_in_bin(current_bin, item):
                bins.append(current_bin)
                current_bin = {'items': [], 'spaces': [(0, 0, self.bin_width, self.bin_height)]}
                if not self.try_pack_in_bin(current_bin, item):
                    return None
        if current_bin['items']:
            bins.append(current_bin)
        return bins

    def try_pack_in_bin(self, bin_dict, item):
        spaces = bin_dict['spaces']
        item_width = item['width']
        item_height = item['height']

        for i, (x, y, w, h) in enumerate(spaces):
            if item_width <= w and item_height <= h:
                del spaces[i]
                if w - item_width > 0:
                    spaces.append((x + item_width, y, w - item_width, item_height))
                if h - item_height > 0:
                    spaces.append((x, y + item_height, w, h - item_height))
                bin_dict['items'].append({
                    'id': item['id'],
                    'x': x,
                    'y': y,
                    'width': item_width,
                    'height': item_height
                })
                return True
        return False

    def compute_fitness(self, solution):
        bins = self.pack_items(solution)
        if bins is None:
            return float('inf')

        total_waste = 0
        for bin_dict in bins:
            used_area = sum(item['width'] * item['height'] for item in bin_dict['items'])
            total_waste += (self.bin_width * self.bin_height) - used_area

        return len(bins) * 1000 + total_waste * 0.1

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores)[:int(ga_choose_ratio * len(scores))]
        return [self.fruits[i] for i in sort_index], [scores[i] for i in sort_index]

    def roulette_selection(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub / sum_score for sub in genes_score]
        rand1, rand2 = np.random.rand(), np.random.rand()
        for i, sub in enumerate(score_ratio):
            rand1 -= sub
            rand2 -= sub
            if rand1 < 0 and rand2 < 0:
                return list(genes_choose[i]), list(genes_choose[i])
            elif rand1 < 0:
                index1 = i
            elif rand2 < 0:
                index2 = i
        return list(genes_choose[index1]), list(genes_choose[index2])

    def tournament_selection(self, genes_score, genes_choose, tournament_size=3):
        population_size = len(genes_score)

        def select_one():
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = [genes_score[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return list(genes_choose[winner_idx])

        parent1 = select_one()
        parent2 = select_one()

        return parent1, parent2

    def order_crossover(self, x, y):
        if len(x) <= 2:
            return x, y

        start, end = sorted(random.sample(range(len(x)), 2))
        tmp = x[start:end]
        x_conflict_index = [y.index(sub) for sub in tmp if y.index(sub) not in range(start, end)]
        y_confict_index = [x.index(sub) for sub in y[start:end] if x.index(sub) not in range(start, end)]
        x[start:end], y[start:end] = y[start:end], tmp
        for i, j in zip(x_conflict_index, y_confict_index):
            y[i], x[j] = x[j], y[i]

        return list(x), list(y)

    def pmx_crossover(self, parent1, parent2):
        if len(parent1) <= 2:
            return parent1, parent2

        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        cx_points = sorted(random.sample(range(len(parent1)), 2))

        mapping1 = {}
        mapping2 = {}

        for i in range(cx_points[0], cx_points[1]):
            mapping1[parent2[i]] = parent1[i]
            mapping2[parent1[i]] = parent2[i]
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]

        for i in list(range(0, cx_points[0])) + list(range(cx_points[1], len(parent1))):
            val1 = parent1[i]
            while val1 in mapping1:
                val1 = mapping1[val1]
            offspring1[i] = val1

            val2 = parent2[i]
            while val2 in mapping2:
                val2 = mapping2[val2]
            offspring2[i] = val2

        return offspring1, offspring2

    def ga_cross(self, x, y):
        if self.crossover_type == 'order':
            return self.order_crossover(x, y)
        else:
            return self.pmx_crossover(x, y)

    def ga_mutate(self, gene):
        if len(gene) <= 2:
            return gene

        start, end = sorted(random.sample(range(len(gene)), 2))

        if self.mutation_type == 'reverse':
            gene[start:end] = gene[start:end][::-1]
        else:
            gene[start], gene[end] = gene[end], gene[start]

        return list(gene)

    def ga(self):
        fitness_scores = np.array([self.compute_fitness(fruit) for fruit in self.fruits])
        normalized_fitness = (np.max(fitness_scores) - fitness_scores) / \
                             (np.max(fitness_scores) - np.min(fitness_scores)) \
            if np.max(fitness_scores) != np.min(fitness_scores) \
            else np.ones_like(fitness_scores)

        parents, parents_score = self.ga_parent(normalized_fitness, self.ga_choose_ratio)

        fitness_values = [self.compute_fitness(parent) for parent in parents]
        best_idx = np.argmin(fitness_values)
        tmp_best_one = parents[best_idx]
        tmp_best_fitness = fitness_values[best_idx]

        fruits = parents.copy()

        while len(fruits) < self.num_total:
            if self.selection_method == 'roulette':
                gene_x, gene_y = self.roulette_selection(parents_score, parents)
            else:
                gene_x, gene_y = self.tournament_selection(parents_score, parents)

            rand = np.random.random()
            if rand < self.crossover_rate:
                gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            elif rand < self.crossover_rate + self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x)
                gene_y_new = self.ga_mutate(gene_y)
            else:
                gene_x_new = gene_x
                gene_y_new = gene_y

            x_fitness = self.compute_fitness(gene_x_new)
            y_fitness = self.compute_fitness(gene_y_new)

            if x_fitness < y_fitness and gene_x_new not in fruits:
                fruits.append(gene_x_new)
            elif gene_y_new not in fruits:
                fruits.append(gene_y_new)

        self.fruits = fruits
        return tmp_best_one, tmp_best_fitness

    def run(self):
        generations_without_improvement = 0
        for i in range(1, self.max_iterations + 1):
            self.generation = i
            tmp_best_one, tmp_best_fitness = self.ga()

            if tmp_best_fitness < self.best_fitness:
                improvement = (self.best_fitness - tmp_best_fitness) / self.best_fitness \
                    if self.best_fitness != float('inf') else 1.0
                self.best_fitness = tmp_best_fitness
                self.best_solution = tmp_best_one
                generations_without_improvement = 0

                if improvement < self.improvement_threshold:
                    generations_without_improvement += 1
            else:
                generations_without_improvement += 1

            self.improvement_curve.append(self.best_fitness)

            if i % 10 == 0:
                print(f"Generation {i}: Best Fitness = {self.best_fitness}")
                current_solution = self.pack_items(tmp_best_one)
                if self.update_callback:
                    self.update_callback(self.improvement_curve, current_solution)
                self.save_generation_image(current_solution, self.improvement_curve)

            if generations_without_improvement >= self.early_stop_generations:
                print(f"Early stopping at generation {i}")
                break

        self.create_gif()

        # 在运行结束时保存最终种群信息
        self.save_final_population(self.run_number)

        return self.pack_items(self.best_solution), self.best_fitness, self.improvement_curve

    def save_final_population(self, run_number):
        """保存最终种群的信息到CSV文件"""
        if not self.save_outputs:
            return
            
        # 计算所有个体的适应度
        population_fitness = [(individual, self.compute_fitness(individual)) 
                            for individual in self.fruits]
        
        # 按适应度排序
        population_fitness.sort(key=lambda x: x[1])
        
        # 准备CSV文件路径
        population_file = os.path.join(self.run_output_dir, 'final_population.csv')
        
        # 写入CSV文件
        with open(population_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Individual_ID', 'Fitness', 'Solution_Sequence'])
            
            # 写入每个个体的信息
            for idx, (solution, fitness) in enumerate(population_fitness, 1):
                writer.writerow([idx, fitness, ','.join(map(str, solution))])

    def create_woc_solution_original(self, all_populations, num_items, top_ratio=0.2, pairs_threshold=0.5):
        """基于多次运行的种群创建WOC解决方案"""
        # 收集所有种群中最优的解
        all_best_solutions = []
        for population in all_populations:
            solutions_with_fitness = [(sol, self.compute_fitness(sol)) for sol in population]
            solutions_with_fitness.sort(key=lambda x: x[1])  # 假设fitness越小越好
            num_to_select = max(1, int(len(population) * top_ratio))  # 确保至少选择一个解
            all_best_solutions.extend([sol for sol, _ in solutions_with_fitness[:num_to_select]])

        # 创建agreement矩阵
        agreement_matrix = np.zeros((num_items, num_items))
        for solution in all_best_solutions:
            for i in range(len(solution) - 1):
                item1, item2 = solution[i], solution[i + 1]
                agreement_matrix[item1][item2] += 1

        # 找出最强的项目对
        pairs = []
        max_agreement = np.max(agreement_matrix)
        if max_agreement > 0:
            threshold = max_agreement * pairs_threshold
            for i in range(num_items):
                for j in range(num_items):
                    if agreement_matrix[i][j] >= threshold:
                        pairs.append((i, j, agreement_matrix[i][j]))

        # 按照agreement值排序pairs
        pairs.sort(key=lambda x: x[2], reverse=True)

        # 初始化解决方案
        solution = []
        used_items = set()

        # 从最强的pairs开始构建解决方案
        for item1, item2, _ in pairs:  # 修复了语法错误
            if len(used_items) >= num_items:  # 修复了变量名错误
                break

            if item1 not in used_items and item2 not in used_items:
                solution.extend([item1, item2])
                used_items.add(item1)
                used_items.add(item2)

            elif item2 not in used_items and len(solution) > 0 and solution[-1] == item1:
                solution.append(item2)
                used_items.add(item2)

        # 使用贪婪算法添加剩余项目
        while len(solution) < num_items:
            current_item = solution[-1] if solution else None

            if current_item is not None:
                row = agreement_matrix[current_item].copy()  # 创建副本避免修改原始数据
                # 将已使用的项设置为-1
                for i in range(num_items):
                    if i in used_items:
                        row[i] = -1

                if np.max(row) > 0:
                    next_item = np.argmax(row)
                    solution.append(next_item)
                    used_items.add(next_item)
                else:
                    # 如果没有更好的选择，选择任意未使用的项
                    unused_items = list(set(range(num_items)) - used_items)
                    if unused_items:
                        next_item = unused_items[0]
                        solution.append(next_item)
                        used_items.add(next_item)
            else:
                # 如果solution为空，选择任意未使用的项
                unused_items = list(set(range(num_items)) - used_items)
                if unused_items:
                    next_item = unused_items[0]
                    solution.append(next_item)
                    used_items.add(next_item)

        return solution

    def create_woc_solution(self, all_populations, num_items, top_ratio=0.2):
        """Improved WOC solution creation"""
        # Collect best solutions
        all_best_solutions = []
        for population in all_populations:
            solutions_with_fitness = [(sol, self.compute_fitness(sol)) for sol in population]
            solutions_with_fitness.sort(key=lambda x: x[1])
            num_to_select = max(1, int(len(population) * top_ratio))
            all_best_solutions.extend([sol for sol, _ in solutions_with_fitness[:num_to_select]])

        # Create enhanced agreement matrix that considers wider context
        agreement_matrix = np.zeros((num_items, num_items))
        for solution in all_best_solutions:
            # Consider items within a window, not just adjacent pairs
            window_size = 3
            for i in range(len(solution)):
                for j in range(1, window_size + 1):
                    if i + j < len(solution):
                        item1, item2 = solution[i], solution[i + j]
                        # Weight decreases with distance
                        agreement_matrix[item1][item2] += 1.0 / j
                        agreement_matrix[item2][item1] += 1.0 / j

        # Generate multiple candidates using probabilistic reconstruction
        num_candidates = 10
        candidates = []

        for _ in range(num_candidates):
            solution = []
            used_items = set()
            current_item = None

            while len(solution) < num_items:
                if not current_item:
                    # Start with random unused item
                    available = list(set(range(num_items)) - used_items)
                    current_item = random.choice(available)
                else:
                    # Probabilistic selection of next item
                    weights = agreement_matrix[current_item].copy()
                    # Zero out used items
                    for used in used_items:
                        weights[used] = 0

                    if np.sum(weights) > 0:
                        # Probabilistic selection weighted by agreement scores
                        probs = weights / np.sum(weights)
                        current_item = np.random.choice(range(num_items), p=probs)
                    else:
                        # If no weights, choose randomly from remaining items
                        available = list(set(range(num_items)) - used_items)
                        current_item = random.choice(available)

                solution.append(current_item)
                used_items.add(current_item)

            candidates.append(solution)

        # Evaluate all candidates and return the best one
        best_fitness = float('inf')
        best_candidate = None

        for candidate in candidates:
            fitness = self.compute_fitness(candidate)
            if fitness < best_fitness:
                best_fitness = fitness
                best_candidate = candidate

        return best_candidate
class BinPackingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Bin Packing Genetic Algorithm")

        self.frame = ttk.Frame(master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)

        self.plot_frame = ttk.Frame(self.frame)
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)

        self.setup_control_panel()
        self.setup_plot_panel()

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.current_run = None
        self.is_running = False

    def setup_control_panel(self):
        controls = [
            ("Bin Width:", "bin_width", "10"),
            ("Bin Height:", "bin_height", "10"),
            ("Mutation Rate:", "mutation_rate", "0.05"),
            ("Crossover Rate:", "crossover_rate", "0.85"),
            ("Population Size:", "population_size", "25"),
            ("Number of Runs:", "num_runs", "10"),
            ("Max Iterations:", "max_iterations", "2000"),
            ("Early Stop Generations:", "early_stop_generations", "200"),
            ("Improvement Threshold:", "improvement_threshold", "0.001"),
            ("Output Folder Name:", "output_folder", "experiment_results"),
            ("WOC Top Ratio:", "woc_top_ratio", "0.2"),
        ]

        for i, (label_text, var_name, default_value) in enumerate(controls):
            ttk.Label(self.control_frame, text=label_text).grid(row=i, column=0, sticky=tk.W)
            setattr(self, var_name, tk.StringVar(value=default_value))
            ttk.Entry(self.control_frame, textvariable=getattr(self, var_name)).grid(
                row=i, column=1, sticky=(tk.W, tk.E))

        row = len(controls)

        # 添加数据集选择框
        ttk.Label(self.control_frame, text="Dataset Path:").grid(row=row, column=0, sticky=tk.W)
        self.dataset_path = tk.StringVar(value="50.json")
        dataset_frame = ttk.Frame(self.control_frame)
        dataset_frame.grid(row=row, column=1, sticky=(tk.W, tk.E))
        ttk.Entry(dataset_frame, textvariable=self.dataset_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).pack(side=tk.RIGHT)
        row += 1

        # Add save outputs checkbox
        self.save_outputs = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Save Images & GIF", variable=self.save_outputs).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        # WOC选择框
        self.use_woc = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Use WOC Algorithm", 
                       variable=self.use_woc).grid(
            row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        # Selection method
        ttk.Label(self.control_frame, text="Selection Method:").grid(row=row, column=0, sticky=tk.W)
        self.selection_method = tk.StringVar(value="roulette")
        selection_combo = ttk.Combobox(self.control_frame, textvariable=self.selection_method)
        selection_combo['values'] = ('tournament', 'roulette')
        selection_combo['state'] = 'readonly'
        selection_combo.grid(row=row, column=1, sticky=(tk.W, tk.E))
        row += 1

        # Mutation type
        ttk.Label(self.control_frame, text="Mutation Type:").grid(row=row, column=0, sticky=tk.W)
        self.mutation_type = tk.StringVar(value="reverse")
        mutation_combo = ttk.Combobox(self.control_frame, textvariable=self.mutation_type)
        mutation_combo['values'] = ('reverse', 'swap')
        mutation_combo['state'] = 'readonly'
        mutation_combo.grid(row=row, column=1, sticky=(tk.W, tk.E))
        row += 1

        # Crossover type
        ttk.Label(self.control_frame, text="Crossover Type:").grid(row=row, column=0, sticky=tk.W)
        self.crossover_type = tk.StringVar(value="order")
        crossover_combo = ttk.Combobox(self.control_frame, textvariable=self.crossover_type)
        crossover_combo['values'] = ('order', 'pmx')
        crossover_combo['state'] = 'readonly'
        crossover_combo.grid(row=row, column=1, sticky=(tk.W, tk.E))
        row += 1

        # 修改按钮布局部分
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.run_button = ttk.Button(button_frame, text="Run GA", command=self.run_ga)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_ga)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button['state'] = 'disabled'
        row += 1

        # Results text
        self.result_text = tk.Text(self.control_frame, height=10, width=40)
        self.result_text.grid(row=row, column=0, columnspan=2)

    def save_run_statistics(self, all_stats, params):
        """保存所有运行的统计数据到单个CSV文件"""
        output_folder = self.output_folder.get()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        filename = os.path.join(output_folder, 'experiment_results.csv')

        # 定义CSV文件的表头
        headers = [
            'run_number',
            'running_time',
            'num_bins',
            'gap_space',
            'fitness',
            'space_utilization',
            'bin_width',
            'bin_height',
            'mutation_rate',
            'crossover_rate',
            'population_size',
            'max_iterations',
            'early_stop_generations',
            'improvement_threshold',
            'selection_method',
            'mutation_type',
            'crossover_type'
        ]

        # 准备每次运行的数据
        rows = []
        for run_num, stats in enumerate(all_stats, 1):
            row = [
                run_num,
                stats['execution_time'],
                stats['num_bins'],
                stats['empty_space'],
                stats['fitness'],
                stats['utilization'],
                params['bin_width'],
                params['bin_height'],
                params['mutation_rate'],
                params['crossover_rate'],
                params['population_size'],
                params['max_iterations'],
                params['early_stop_generations'],
                params['improvement_threshold'],
                self.selection_method.get(),
                self.mutation_type.get(),
                self.crossover_type.get()
            ]
            rows.append(row)

        # 写入CSV文件
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入表头
            writer.writerows(rows)  # 写入所有数据行

    def setup_plot_panel(self):
        self.fig = plt.figure(figsize=(8, 10))
        self.ax2 = self.fig.add_axes([0.1, 0.35, 0.8, 0.6])
        self.ax1 = self.fig.add_axes([0.1, 0.05, 0.8, 0.25])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0)

    def update_plots(self, improvement_curve, bins):
        if not self.is_running:
            return

        self.ax1.clear()
        self.ax1.plot(improvement_curve)
        self.ax1.set_title('Fitness Improvement')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.grid(True)

        self.ax2.clear()
        colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, 300))
        bin_width = float(self.bin_width.get())
        bin_height = float(self.bin_height.get())

        bins_per_row = 4
        spacing = 2
        grid_width = bins_per_row * (bin_width + spacing)
        num_rows = math.ceil(len(bins) / bins_per_row)

        for i, bin_dict in enumerate(bins):
            row = i // bins_per_row
            col = i % bins_per_row

            x_pos = col * (bin_width + spacing)
            y_pos = (num_rows - 1 - row) * (bin_height + spacing)

            bin_rect = plt.Rectangle((x_pos, y_pos), bin_width, bin_height,
                                   facecolor='none', edgecolor='black', linestyle='--')
            self.ax2.add_patch(bin_rect)

            for item in bin_dict['items']:
                item_x = x_pos + item['x']
                item_y = y_pos + item['y']
                item_w = item['width']
                item_h = item['height']
                color = colors[item['id'] - 1]

                rect = plt.Rectangle((item_x, item_y), item_w, item_h,
                                   facecolor=color, edgecolor='black', alpha=0.7)
                self.ax2.add_patch(rect)
                self.ax2.text(item_x + item_w / 2, item_y + item_h / 2, f'{item["id"]}',
                             ha='center', va='center')

        self.ax2.set_xlim(-spacing, grid_width + spacing)
        self.ax2.set_ylim(-spacing, num_rows * (bin_height + spacing) + spacing)
        self.ax2.set_title(f'Current Best Solution (Generation {len(improvement_curve)})')
        self.ax2.set_aspect('equal')
        self.ax2.grid(True)

        self.canvas.draw()
        self.master.update()

    def stop_ga(self):
        self.is_running = False
        self.run_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'
        self.status_var.set("Stopped")

    def run_ga(self):
        if self.is_running:
            return

        self.is_running = True
        self.run_button['state'] = 'disabled'
        self.stop_button['state'] = 'normal'
        self.status_var.set("Running...")

        mutation_rate = float(self.mutation_rate.get())
        crossover_rate = float(self.crossover_rate.get())

        if mutation_rate + crossover_rate > 1:
            self.status_var.set("Error: Mutation rate + Crossover rate must be <= 1")
            self.stop_ga()
            return

        params = {
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'population_size': int(self.population_size.get()),
            'num_runs': int(self.num_runs.get()),
            'max_iterations': int(self.max_iterations.get()),
            'early_stop_generations': int(self.early_stop_generations.get()),
            'improvement_threshold': float(self.improvement_threshold.get()),
            'bin_width': float(self.bin_width.get()),
            'bin_height': float(self.bin_height.get()),
            'save_outputs': self.save_outputs.get()
        }

        self.master.after(100, self.run_ga_iteration, params)

    def run_ga_iteration(self, params):
        if not self.is_running:
            return

        try:
            items = load_json_data(self.dataset_path.get())
            self.status_var.set(f"successfully loaded {len(items)} items")
        except Exception as e:
            self.status_var.set(f"error: {str(e)}")
            self.stop_ga()
            return

        best_solutions = []
        best_fitnesses = []
        execution_times = []
        empty_spaces = []
        all_stats = []
        all_populations = []

        output_folder = self.output_folder.get()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for run in range(params['num_runs']):
            if not self.is_running:
                break

            start_time = time.time()
            model = GA(
                num_items=len(items),
                num_total=params['population_size'],
                max_iterations=params['max_iterations'],
                items=items,
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate'],
                population_size=params['population_size'],
                early_stop_generations=params['early_stop_generations'],
                improvement_threshold=params['improvement_threshold'],
                bin_width=params['bin_width'],
                bin_height=params['bin_height'],
                update_callback=self.update_plots,
                selection_method=self.selection_method.get(),
                mutation_type=self.mutation_type.get(),
                crossover_type=self.crossover_type.get(),
                save_outputs=params['save_outputs'],
                run_number=run + 1,
                output_folder=output_folder
            )

            solution, fitness, improvement_curve = model.run()
            if not self.is_running:
                break

            execution_time = time.time() - start_time

            total_bin_area = params['bin_width'] * params['bin_height'] * len(solution)
            used_area = sum(sum(item['width'] * item['height'] for item in bin_dict['items']) for bin_dict in solution)
            empty_space = total_bin_area - used_area
            utilization = ((total_bin_area - empty_space) / total_bin_area) * 100

            stats = {
                'execution_time': execution_time,
                'num_bins': len(solution),
                'empty_space': empty_space,
                'fitness': fitness,
                'utilization': utilization
            }
            all_stats.append(stats)

            best_solutions.append(solution)
            best_fitnesses.append(fitness)
            execution_times.append(execution_time)
            empty_spaces.append(empty_space)

            run_results_file = os.path.join(output_folder, f'run_{run + 1}', 'run_results.txt')
            os.makedirs(os.path.dirname(run_results_file), exist_ok=True)
            with open(run_results_file, 'w') as f:
                f.write(f"""Run {run + 1} Results:
    Number of Bins: {len(solution)}
    Fitness: {fitness:.2f}
    Empty Space: {empty_space:.2f} square units
    Space Utilization: {utilization:.2f}%
    Execution Time: {execution_time:.2f} seconds
    """)

            self.status_var.set(f"Completed run {run + 1} of {params['num_runs']}")
            all_populations.append(model.fruits)

        if self.is_running and self.use_woc.get() and len(all_populations) > 0:
            # Create WOC solution
            woc_solution = model.create_woc_solution(
                all_populations,
                len(items),
                top_ratio=float(self.woc_top_ratio.get())
            )

            woc_bins = model.pack_items(woc_solution)
            woc_fitness = model.compute_fitness(woc_solution)

            # Save WOC visualization
            woc_output_dir = os.path.join(output_folder, 'woc_results')
            os.makedirs(woc_output_dir, exist_ok=True)

            # Create figure for WOC solution
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # Plot WOC solution
            colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, 300))
            bin_width = float(self.bin_width.get())
            bin_height = float(self.bin_height.get())

            bins_per_row = 4
            spacing = 2
            grid_width = bins_per_row * (bin_width + spacing)
            num_rows = math.ceil(len(woc_bins) / bins_per_row)

            for i, bin_dict in enumerate(woc_bins):
                row = i // bins_per_row
                col = i % bins_per_row

                x_pos = col * (bin_width + spacing)
                y_pos = (num_rows - 1 - row) * (bin_height + spacing)

                # Draw bin
                bin_rect = plt.Rectangle((x_pos, y_pos), bin_width, bin_height,
                                         facecolor='none', edgecolor='black', linestyle='--')
                ax.add_patch(bin_rect)

                # Draw items in bin
                for item in bin_dict['items']:
                    item_x = x_pos + item['x']
                    item_y = y_pos + item['y']
                    item_w = item['width']
                    item_h = item['height']
                    color = colors[item['id'] - 1]

                    rect = plt.Rectangle((item_x, item_y), item_w, item_h,
                                         facecolor=color, edgecolor='black', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(item_x + item_w / 2, item_y + item_h / 2, f'{item["id"]}',
                            ha='center', va='center')

            ax.set_xlim(-spacing, grid_width + spacing)
            ax.set_ylim(-spacing, num_rows * (bin_height + spacing) + spacing)
            ax.set_title('WOC Solution Visualization')
            ax.set_aspect('equal')
            ax.grid(True)

            # Save WOC visualization
            plt.savefig(os.path.join(woc_output_dir, 'woc_solution.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Update current display
            self.update_plots(improvement_curve, woc_bins)

            best_solutions.append(woc_bins)
            best_fitnesses.append(woc_fitness)

            # Calculate WOC statistics
            total_bin_area = params['bin_width'] * params['bin_height'] * len(woc_bins)
            used_area = sum(sum(item['width'] * item['height'] for item in bin_dict['items']) for bin_dict in woc_bins)
            woc_empty_space = total_bin_area - used_area
            woc_utilization = ((total_bin_area - woc_empty_space) / total_bin_area) * 100

            # Save WOC results
            woc_results_file = os.path.join(woc_output_dir, 'woc_results.txt')
            with open(woc_results_file, 'w') as f:
                f.write(f"""WOC Algorithm Results:
    Number of Bins: {len(woc_bins)}
    Fitness: {woc_fitness:.2f}
    Empty Space: {woc_empty_space:.2f} square units
    Space Utilization: {woc_utilization:.2f}%
    Solution Sequence: {','.join(map(str, woc_solution))}
    """)

        if self.is_running:
            self.save_run_statistics(all_stats, params)

            best_run_idx = np.argmin(best_fitnesses)
            best_solution = best_solutions[best_run_idx]
            best_fitness = best_fitnesses[best_run_idx]
            best_empty_space = empty_spaces[best_run_idx]

            total_area = params['bin_width'] * params['bin_height'] * len(best_solution)
            utilization = ((total_area - best_empty_space) / total_area) * 100

            summary_file = os.path.join(output_folder, 'summary_results.txt')
            with open(summary_file, 'w') as f:
                summary_str = f"""Overall Results Summary:
    Best Run: Run {best_run_idx + 1}
    Number of Bins Used: {len(best_solution)}
    Best Fitness: {best_fitness:.2f}
    Average Fitness: {np.mean(best_fitnesses):.2f}
    Fitness Std Dev: {np.std(best_fitnesses):.2f}
    Empty Space: {best_empty_space:.2f} square units
    Space Utilization: {utilization:.2f}%
    Average Time per Run: {np.mean(execution_times):.2f} seconds
    Results saved in: {output_folder}
    """
                f.write(summary_str)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, summary_str)

        self.is_running = False
        self.run_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'
        self.status_var.set("Completed")

    def browse_dataset(self):
        """打开文件选择对话框选择数据集"""
        filename = tk.filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            self.dataset_path.set(filename)

def load_json_data(file_path):
    """从JSON文件加载数据并转换为所需格式"""
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    # 转换为遗传算法所需的格式
    items = []
    for item in json_data:
        items.append({
            'id': item['id'],
            'width': item['width'],
            'height': item['height'],
            'area': item['width'] * item['height']  # 添加面积计算
        })
    return items

def main():
    root = tk.Tk()
    gui = BinPackingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
