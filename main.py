import random
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
from items import get_test_data


class GA:
    def __init__(self, num_items, num_total, max_iterations, items, mutation_rate=0.05, population_size=25,
                 early_stop_generations=200, improvement_threshold=0.01, woc_experts_ratio=0.2,
                 agreement_weight=0.3, b1=3, b2=3, use_woc=True, bin_width=10, bin_height=10,
                 update_callback=None):
        self.num_items = num_items
        self.num_total = num_total
        self.max_iterations = max_iterations
        self.items = items
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = mutation_rate
        self.population_size = population_size
        self.early_stop_generations = early_stop_generations
        self.improvement_threshold = improvement_threshold
        self.woc_experts_ratio = woc_experts_ratio
        self.agreement_weight = agreement_weight
        self.b1 = b1
        self.b2 = b2
        self.use_woc = use_woc
        self.best_fitness = float('inf')
        self.best_solution = None
        self.improvement_curve = []
        self.agreement_matrix = np.zeros((num_items, num_items))
        self.fruits = self.init_population()
        self.update_callback = update_callback

    def init_population(self):
        population = []
        for _ in range(self.num_total):
            solution = list(range(self.num_items))
            random.shuffle(solution)
            population.append(solution)
        return population

    def compute_agreement_matrix(self, solutions):
        n = self.num_items
        agreement_matrix = np.zeros((n, n))
        num_solutions = len(solutions)
        for solution in solutions:
            for i in range(len(solution) - 1):
                item1, item2 = solution[i], solution[i + 1]
                agreement_matrix[item1][item2] += 1
                agreement_matrix[item2][item1] += 1
        agreement_matrix /= num_solutions
        agreement_matrix = np.power(agreement_matrix, self.b1)
        return agreement_matrix

    def update_agreement_matrix(self):
        scores = self.compute_adp(self.fruits)
        num_experts = int(len(self.fruits) * self.woc_experts_ratio)
        expert_indices = np.argsort(-scores)[:num_experts]
        expert_solutions = [self.fruits[i] for i in expert_indices]
        self.agreement_matrix = self.compute_agreement_matrix(expert_solutions)

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

    def compute_adp(self, fruits):
        fitness_scores = np.array([self.compute_fitness(fruit) for fruit in fruits])

        if not self.use_woc:
            # If WOC is disabled, only use fitness scores
            normalized_fitness = (np.max(fitness_scores) - fitness_scores) / \
                                 (np.max(fitness_scores) - np.min(fitness_scores)) \
                if np.max(fitness_scores) != np.min(fitness_scores) \
                else np.ones_like(fitness_scores)
            return normalized_fitness

        # If WOC is enabled, include agreement scores
        agreement_scores = []
        for fruit in fruits:
            score = 0
            for i in range(len(fruit) - 1):
                score += self.agreement_matrix[fruit[i]][fruit[i + 1]]
            agreement_scores.append(score / (len(fruit) - 1))
        agreement_scores = np.array(agreement_scores)

        normalized_fitness = (np.max(fitness_scores) - fitness_scores) / \
                             (np.max(fitness_scores) - np.min(fitness_scores)) \
            if np.max(fitness_scores) != np.min(fitness_scores) \
            else np.ones_like(fitness_scores)

        if len(set(agreement_scores)) == 1:
            normalized_agreement = np.ones_like(agreement_scores)
        else:
            normalized_agreement = (agreement_scores - np.min(agreement_scores)) / \
                                   (np.max(agreement_scores) - np.min(agreement_scores))

        return (1 - self.agreement_weight) * normalized_fitness + \
            self.agreement_weight * normalized_agreement

    def ga_cross(self, x, y):
        if len(x) <= 2:
            return x, y

        segment_scores = []
        for i in range(len(x) - 1):
            score = self.agreement_matrix[x[i]][x[i + 1]]
            segment_scores.append(score)

        segment_scores = np.array(segment_scores)
        probabilities = 1 - segment_scores / np.sum(segment_scores)
        probabilities = probabilities / np.sum(probabilities)

        try:
            crossover_points = np.random.choice(len(x) - 1, size=2, replace=False, p=probabilities)
            start, end = sorted(crossover_points)
        except:
            order = sorted(random.sample(range(len(x)), 2))
            start, end = order

        tmp = x[start:end]
        x_conflict_index = [y.index(sub) for sub in tmp if y.index(sub) not in range(start, end)]
        y_confict_index = [x.index(sub) for sub in y[start:end] if x.index(sub) not in range(start, end)]
        x[start:end], y[start:end] = y[start:end], tmp
        for i, j in zip(x_conflict_index, y_confict_index):
            y[i], x[j] = x[j], y[i]

        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores)[:int(ga_choose_ratio * len(scores))]
        return [self.fruits[i] for i in sort_index], [scores[i] for i in sort_index]

    def ga_choose(self, genes_score, genes_choose):
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

    def ga_mutate(self, gene):
        if len(gene) <= 2:
            return gene

        segment_scores = []
        for i in range(len(gene) - 1):
            score = self.agreement_matrix[gene[i]][gene[i + 1]]
            segment_scores.append(score)

        segment_scores = np.array(segment_scores)
        probabilities = 1 - segment_scores / np.sum(segment_scores)
        probabilities = probabilities / np.sum(probabilities)

        try:
            mutation_points = np.random.choice(len(gene) - 1, size=2, replace=False, p=probabilities)
            start, end = sorted(mutation_points)
        except:
            start, end = sorted(random.sample(range(len(gene)), 2))

        gene[start:end] = gene[start:end][::-1]
        return list(gene)

    def ga(self):
        if self.use_woc:
            self.update_agreement_matrix()
        scores = self.compute_adp(self.fruits)
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)

        fitness_values = [self.compute_fitness(parent) for parent in parents]
        best_idx = np.argmin(fitness_values)
        tmp_best_one = parents[best_idx]
        tmp_best_fitness = fitness_values[best_idx]

        fruits = parents.copy()

        while len(fruits) < self.num_total:
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)

            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)

            x_fitness = self.compute_fitness(gene_x_new)
            y_fitness = self.compute_fitness(gene_y_new)

            if not self.use_woc:
                # If WOC is disabled, only use fitness for comparison
                if x_fitness < y_fitness and gene_x_new not in fruits:
                    fruits.append(gene_x_new)
                elif gene_y_new not in fruits:
                    fruits.append(gene_y_new)
            else:
                # If WOC is enabled, use both fitness and agreement
                x_agreement = self.compute_agreement_matrix([gene_x_new]).mean()
                y_agreement = self.compute_agreement_matrix([gene_y_new]).mean()

                fitness_scores = np.array([x_fitness, y_fitness])
                normalized_fitness = (np.max(fitness_scores) - fitness_scores) / \
                                     (np.max(fitness_scores) - np.min(fitness_scores)) \
                    if np.max(fitness_scores) != np.min(fitness_scores) \
                    else np.ones(2)

                agreement_scores = np.array([x_agreement, y_agreement])
                normalized_agreement = (agreement_scores - np.min(agreement_scores)) / \
                                       (np.max(agreement_scores) - np.min(agreement_scores)) \
                    if np.max(agreement_scores) != np.min(agreement_scores) \
                    else np.ones(2)

                combined_scores = (1 - self.agreement_weight) * normalized_fitness + \
                                  self.agreement_weight * normalized_agreement

                if combined_scores[0] > combined_scores[1] and gene_x_new not in fruits:
                    fruits.append(gene_x_new)
                elif gene_y_new not in fruits:
                    fruits.append(gene_y_new)

        self.fruits = fruits
        return tmp_best_one, tmp_best_fitness

    def run(self):
        generations_without_improvement = 0
        for i in range(1, self.max_iterations + 1):
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

            # Call the update callback every 10 generations
            if i % 10 == 0:
                print(f"Generation {i}: Best Fitness = {self.best_fitness}")
                if self.update_callback:
                    self.update_callback(self.improvement_curve, self.pack_items(tmp_best_one))

            if generations_without_improvement >= self.early_stop_generations:
                print(f"Early stopping at generation {i}")
                break

        return self.pack_items(self.best_solution), self.best_fitness, self.improvement_curve


class BinPackingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Bin Packing Genetic Algorithm with Wisdom of Crowds")

        # 创建主Frame
        self.frame = ttk.Frame(master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 创建左侧控制面板Frame
        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)

        # 创建右侧图表Frame
        self.plot_frame = ttk.Frame(self.frame)
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)

        self.setup_control_panel()
        self.setup_plot_panel()

        # 添加状态显示
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # 存储当前运行的任务信息
        self.current_run = None
        self.is_running = False

    def setup_control_panel(self):
        # 添加控制面板的组件
        controls = [
            ("Bin Width:", "bin_width", "10"),
            ("Bin Height:", "bin_height", "10"),
            ("Mutation Rate:", "mutation_rate", "0.05"),
            ("Population Size:", "population_size", "25"),
            ("Number of Runs:", "num_runs", "10"),
            ("Max Iterations:", "max_iterations", "2000"),
            ("Early Stop Generations:", "early_stop_generations", "200"),
            ("Improvement Threshold:", "improvement_threshold", "0.01"),
            ("WOC Experts Ratio:", "woc_experts_ratio", "0.2"),
            ("Agreement Weight:", "agreement_weight", "0.3"),
            ("B1 Parameter:", "b1", "3"),
            ("B2 Parameter:", "b2", "3"),
        ]

        for i, (label_text, var_name, default_value) in enumerate(controls):
            ttk.Label(self.control_frame, text=label_text).grid(row=i, column=0, sticky=tk.W)
            setattr(self, var_name, tk.StringVar(value=default_value))
            ttk.Entry(self.control_frame, textvariable=getattr(self, var_name)).grid(
                row=i, column=1, sticky=(tk.W, tk.E))

        # Add WOC toggle
        self.use_woc = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Use Wisdom of Crowds",
                        variable=self.use_woc).grid(row=len(controls), column=0, columnspan=2)

        # Add Run/Stop buttons
        self.run_button = ttk.Button(self.control_frame, text="Run GA", command=self.run_ga)
        self.run_button.grid(row=len(controls) + 1, column=0)

        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_ga)
        self.stop_button.grid(row=len(controls) + 1, column=1)
        self.stop_button['state'] = 'disabled'

        # Add result text
        self.result_text = tk.Text(self.control_frame, height=10, width=40)
        self.result_text.grid(row=len(controls) + 2, column=0, columnspan=2)

    def setup_plot_panel(self):
        self.fig = plt.figure(figsize=(8, 10))
        # Create subplot for bin packing visualization (taking up 70% of vertical space)
        self.ax2 = self.fig.add_axes([0.1, 0.35, 0.8, 0.6])
        # Create subplot for convergence plot (taking up 25% of vertical space)
        self.ax1 = self.fig.add_axes([0.1, 0.05, 0.8, 0.25])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0)

    def update_plots(self, improvement_curve, bins):
        if not self.is_running:
            return

        # Update convergence plot
        self.ax1.clear()
        self.ax1.plot(improvement_curve)
        self.ax1.set_title('Fitness Improvement')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.grid(True)

        # Update bin packing visualization
        self.ax2.clear()
        colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, 30))
        bin_width = float(self.bin_width.get())
        bin_height = float(self.bin_height.get())

        # Calculate grid layout parameters
        bins_per_row = 4
        spacing = 2  # Space between bins
        grid_width = bins_per_row * (bin_width + spacing)
        num_rows = math.ceil(len(bins) / bins_per_row)

        for i, bin_dict in enumerate(bins):
            # Calculate grid position
            row = i // bins_per_row
            col = i % bins_per_row

            # Calculate bin position
            x_pos = col * (bin_width + spacing)
            y_pos = (num_rows - 1 - row) * (bin_height + spacing)  # Reverse row order for bottom-up layout

            # Draw bin outline
            bin_rect = plt.Rectangle((x_pos, y_pos), bin_width, bin_height,
                                     facecolor='none', edgecolor='black', linestyle='--')
            self.ax2.add_patch(bin_rect)

            # Draw items in bin
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

        # Set plot limits to accommodate all bins with spacing
        self.ax2.set_xlim(-spacing, grid_width + spacing)
        self.ax2.set_ylim(-spacing, num_rows * (bin_height + spacing) + spacing)
        self.ax2.set_title('Current Best Solution')
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

        # Get all parameters
        params = {
            'mutation_rate': float(self.mutation_rate.get()),
            'population_size': int(self.population_size.get()),
            'num_runs': int(self.num_runs.get()),
            'max_iterations': int(self.max_iterations.get()),
            'early_stop_generations': int(self.early_stop_generations.get()),
            'improvement_threshold': float(self.improvement_threshold.get()),
            'woc_experts_ratio': float(self.woc_experts_ratio.get()),
            'agreement_weight': float(self.agreement_weight.get()),
            'b1': float(self.b1.get()),
            'b2': float(self.b2.get()),
            'bin_width': float(self.bin_width.get()),
            'bin_height': float(self.bin_height.get()),
            'use_woc': self.use_woc.get()
        }

        # Start GA in a separate thread
        self.master.after(100, self.run_ga_iteration, params)

    def run_ga_iteration(self, params):
        if not self.is_running:
            return

        items = get_test_data()
        best_solutions = []
        best_fitnesses = []
        execution_times = []

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
                population_size=params['population_size'],
                early_stop_generations=params['early_stop_generations'],
                improvement_threshold=params['improvement_threshold'],
                woc_experts_ratio=params['woc_experts_ratio'],
                agreement_weight=params['agreement_weight'],
                b1=params['b1'],
                b2=params['b2'],
                use_woc=params['use_woc'],
                bin_width=params['bin_width'],
                bin_height=params['bin_height'],
                update_callback=self.update_plots
            )

            solution, fitness, improvement_curve = model.run()
            if not self.is_running:
                break

            end_time = time.time()
            best_solutions.append(solution)
            best_fitnesses.append(fitness)
            execution_times.append(end_time - start_time)

            self.status_var.set(f"Completed run {run + 1} of {params['num_runs']}")

        if self.is_running:
            best_run_idx = np.argmin(best_fitnesses)
            best_solution = best_solutions[best_run_idx]
            best_fitness = best_fitnesses[best_run_idx]

            woc_status = "Enabled" if params['use_woc'] else "Disabled"
            result_str = f"""
            Performance Statistics:
            Number of Bins Used: {len(best_solution)}
            Best Fitness: {best_fitness:.2f}
            Average Fitness: {np.mean(best_fitnesses):.2f}
            Fitness Std Dev: {np.std(best_fitnesses):.2f}
            Average Time per Run: {np.mean(execution_times):.2f} seconds
            Wisdom of Crowds: {woc_status}
            Bin Dimensions: {params['bin_width']} x {params['bin_height']}
            """
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_str)

        self.is_running = False
        self.run_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'
        self.status_var.set("Completed")


def main():
    root = tk.Tk()
    gui = BinPackingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()