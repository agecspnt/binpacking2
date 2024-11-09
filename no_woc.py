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


def save_population_agreement_matrix(population, num_items, filename='population_agreement_matrix.txt'):
    """
    Generate and save agreement matrix based on final population

    Args:
        population: List of solutions from final generation
        num_items: Number of total items
        filename: Output file name
    """
    # Initialize agreement matrix
    agreement_matrix = np.zeros((num_items, num_items))

    # Count agreements across all solutions in population
    for solution in population:
        for i in range(len(solution) - 1):
            item1, item2 = solution[i], solution[i + 1]
            agreement_matrix[item1][item2] += 1
            agreement_matrix[item2][item1] += 1

    # # Normalize by number of solutions
    # agreement_matrix = agreement_matrix / len(population)

    # Format matrix as string
    matrix_str = "Population Agreement Matrix (Final Generation):\n"
    matrix_str += f"Population Size: {len(population)}\n"
    matrix_str += "-" * (num_items * 8) + "\n"

    # Add column headers
    matrix_str += "     "  # Space for row headers
    for i in range(num_items):
        matrix_str += f"{i:6d} "
    matrix_str += "\n"
    matrix_str += "-" * (num_items * 8) + "\n"

    # Add matrix content with row headers
    for i in range(num_items):
        matrix_str += f"{i:3d} |"
        for j in range(num_items):
            matrix_str += f"{agreement_matrix[i][j]:6.3f} "
        matrix_str += "\n"

    # Save to file
    with open(filename, 'w') as f:
        f.write(matrix_str)

    return agreement_matrix

class GA:
    def __init__(self, num_items, num_total, max_iterations, items, mutation_rate=0.05, population_size=25,
                 early_stop_generations=200, improvement_threshold=0.01, bin_width=10, bin_height=10,
                 update_callback=None, selection_method='roulette'):
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
        self.best_fitness = float('inf')
        self.best_solution = None
        self.improvement_curve = []
        self.fruits = self.init_population()
        self.update_callback = update_callback
        self.selection_method = selection_method

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

        return len(bins) * 1000 + total_waste * 0.1, total_waste

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores)[:int(ga_choose_ratio * len(scores))]
        return [self.fruits[i] for i in sort_index], [scores[i] for i in sort_index]

    def roulette_selection(self, genes_score, genes_choose):
        # 原来的ga_choose方法代码
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

    def ga_cross(self, x, y):
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

    def ga_mutate(self, gene):
        if len(gene) <= 2:
            return gene

        start, end = sorted(random.sample(range(len(gene)), 2))
        gene[start:end] = gene[start:end][::-1]
        return list(gene)

    def ga(self):
        fitness_scores, ignore_spaces = zip(*[self.compute_fitness(fruit) for fruit in self.fruits])
        fitness_scores = np.array(fitness_scores)
        ignore_spaces = np.array(ignore_spaces)
        normalized_fitness = (np.max(fitness_scores) - fitness_scores) / \
                             (np.max(fitness_scores) - np.min(fitness_scores)) \
            if np.max(fitness_scores) != np.min(fitness_scores) \
            else np.ones_like(fitness_scores)

        parents, parents_score = self.ga_parent(normalized_fitness, self.ga_choose_ratio)

        fitness_values, parent_spaces = zip(*[self.compute_fitness(parent) for parent in parents])
        fitness_values = np.array(fitness_values)
        parent_spaces = np.array(parent_spaces)
        best_idx = np.argmin(fitness_values)
        tmp_best_one = parents[best_idx]
        tmp_best_fitness = fitness_values[best_idx]
        best_gap_space = parent_spaces[best_idx]

        fruits = parents.copy()

        while len(fruits) < self.num_total:
            if self.selection_method == 'roulette':
                gene_x, gene_y = self.roulette_selection(parents_score, parents)
            else:  # tournament
                gene_x, gene_y = self.tournament_selection(parents_score, parents)
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)

            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)

            x_fitness, _ = self.compute_fitness(gene_x_new)
            y_fitness, _ = self.compute_fitness(gene_y_new)

            if x_fitness < y_fitness and gene_x_new not in fruits:
                fruits.append(gene_x_new)
            elif gene_y_new not in fruits:
                fruits.append(gene_y_new)

        self.fruits = fruits
        return tmp_best_one, tmp_best_fitness, best_gap_space

    def run(self):
        generations_without_improvement = 0
        gap_spaces = []
        for i in range(1, self.max_iterations + 1):
            tmp_best_one, tmp_best_fitness, space = self.ga()
            gap_spaces.append(space)

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
                if self.update_callback:
                    self.update_callback(self.improvement_curve, self.pack_items(tmp_best_one))

            if generations_without_improvement >= self.early_stop_generations:
                print(f"Early stopping at generation {i}")
                break

        # 在结束前保存最后一代的协议矩阵
        save_population_agreement_matrix(self.fruits, self.num_items)



        return self.pack_items(self.best_solution), self.best_fitness, self.improvement_curve, gap_spaces


class BinPackingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Bin Packing Genetic Algorithm")

        # Create main Frame
        self.frame = ttk.Frame(master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create left control panel Frame
        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)

        # Create right chart Frame
        self.plot_frame = ttk.Frame(self.frame)
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)

        self.setup_control_panel()
        self.setup_plot_panel()

        # Add status display
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Store current task information
        self.current_run = None
        self.is_running = False

    def setup_control_panel(self):
        # Add control panel components
        controls = [
            ("Bin Width:", "bin_width", "10"),
            ("Bin Height:", "bin_height", "10"),
            ("Mutation Rate:", "mutation_rate", "0.05"),
            ("Population Size:", "population_size", "25"),
            ("Number of Runs:", "num_runs", "10"),
            ("Max Iterations:", "max_iterations", "2000"),
            ("Early Stop Generations:", "early_stop_generations", "200"),
            ("Improvement Threshold:", "improvement_threshold", "0.01"),
        ]

        for i, (label_text, var_name, default_value) in enumerate(controls):
            ttk.Label(self.control_frame, text=label_text).grid(row=i, column=0, sticky=tk.W)
            setattr(self, var_name, tk.StringVar(value=default_value))
            ttk.Entry(self.control_frame, textvariable=getattr(self, var_name)).grid(
                row=i, column=1, sticky=(tk.W, tk.E))

        # Add Run/Stop buttons
        self.run_button = ttk.Button(self.control_frame, text="Run GA", command=self.run_ga)
        self.run_button.grid(row=len(controls), column=0)

        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_ga)
        self.stop_button.grid(row=len(controls), column=1)
        self.stop_button['state'] = 'disabled'

        # Add result text
        self.result_text = tk.Text(self.control_frame, height=10, width=40)
        self.result_text.grid(row=len(controls) + 1, column=0, columnspan=2)

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
            'bin_width': float(self.bin_width.get()),
            'bin_height': float(self.bin_height.get()),
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
        total_gap_spaces = []

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
                bin_width=params['bin_width'],
                bin_height=params['bin_height'],
                update_callback=self.update_plots
            )

            solution, fitness, improvement_curve, gap_spaces = model.run()
            if not self.is_running:
                break

            end_time = time.time()
            best_solutions.append(solution)
            best_fitnesses.append(fitness)
            execution_times.append(end_time - start_time)
            total_gap_spaces.append(gap_spaces)

            self.status_var.set(f"Completed run {run + 1} of {params['num_runs']}")

        if self.is_running:
            best_run_idx = np.argmin(best_fitnesses)
            best_solution = best_solutions[best_run_idx]
            best_fitness = best_fitnesses[best_run_idx]

            result_str = f"""
            Performance Statistics:
            Number of Bins Used: {len(best_solution)}
            Best Fitness: {best_fitness:.2f}
            Average Fitness: {np.mean(best_fitnesses):.2f}
            Fitness Std Dev: {np.std(best_fitnesses):.2f}
            Average Time per Run: {np.mean(execution_times):.2f} seconds
            Bin Dimensions: {params['bin_width']} x {params['bin_height']}
            """
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_str)

        self.is_running = False
        self.run_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'
        self.status_var.set("Completed")

        #Saving run data to csv file
        with open('runDataFile.csv',mode='w') as runDataFile:
            run_data_file_writer = csv.writer(runDataFile,delimiter=',')
            run_data_file_writer.writerow(['Running Time','Number of Bins','Gap Space'])
            spaces_text = []
            for spaces in total_gap_spaces:
                spaces_text_line = "["
                for i in range(len(spaces)):
                    if i < len(spaces) - 1:
                        spaces_text_line += (str(spaces[i]) + ", ")
                    else:
                        spaces_text_line += str(spaces[i])
                spaces_text.append(spaces_text_line)
            for i in range(len(best_solutions)):
                run_data_file_writer.writerow([str(execution_times[i]),str(best_solutions[i]),str(spaces_text[i])])


def main():
    root = tk.Tk()
    gui = BinPackingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()







