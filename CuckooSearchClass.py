import numpy as np
import math
from sklearn.model_selection import cross_val_score

class CuckooSearch:
    def __init__(self, fitness_func, bounds, n_nests=10, n_iter=20, pa=0.25):
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.n_nests = n_nests
        self.n_iter = n_iter
        self.pa = pa  # probability of abandoning nests

    def initialize_nests(self):
        nests = []
        for _ in range(self.n_nests):
            nest = []
            for (low, high) in self.bounds:
                nest.append(np.random.uniform(low, high))
            nests.append(nest)
        return np.array(nests)

    def simple_bounds(self, solution):
        solution_new = []
        for i, val in enumerate(solution):
            low, high = self.bounds[i]
            solution_new.append(np.clip(val, low, high))
        return np.array(solution_new)

    def get_best(self, nests):
        fitness = [self.fitness_func(nest) for nest in nests]
        idx = np.argmin(fitness)
        return nests[idx], fitness[idx]

    def levy_flight(self, solution):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.randn(len(solution)) * sigma
        v = np.random.randn(len(solution))
        step = u / (np.abs(v) ** (1 / beta))

        return solution + 0.01 * step

    def run(self):
        nests = self.initialize_nests()
        best_solution, best_fitness = self.get_best(nests)

        for iteration in range(self.n_iter):
            print(f"Iteration {iteration+1}/{self.n_iter}...")


            for i in range(self.n_nests):
                new_solution = self.levy_flight(nests[i])
                new_solution = self.simple_bounds(new_solution)

                if self.fitness_func(new_solution) < self.fitness_func(nests[i]):
                    nests[i] = new_solution

            for i in range(self.n_nests):
                if np.random.rand() < self.pa:
                    nests[i] = self.initialize_nests()[0]

            current_best, current_fitness = self.get_best(nests)

            if current_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_fitness

            print("Best so far:", best_solution, "F1:", -best_fitness)

        return best_solution, best_fitness
