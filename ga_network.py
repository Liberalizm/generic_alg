import math
import random
import os
import pickle
from typing import List, Callable, Optional

from cat import NeuralNetworkInterface


class Gene:
    """
    Base gene class. Concrete genes encode layer sizes and weights.
    """


class LayerGene(Gene):
    def __init__(self, in_size: int, out_size: int):
        self.in_size = in_size
        self.out_size = out_size


class WeightsGene(Gene):
    def __init__(self, rows: int, cols: int, init_scale: float = 0.5):
        # Matrix shape: rows x cols, stored row-major
        self.rows = rows
        self.cols = cols
        self.values = [random.uniform(-init_scale, init_scale) for _ in range(rows * cols)]

    def get(self, r: int, c: int) -> float:
        return self.values[r * self.cols + c]

    def set(self, r: int, c: int, v: float):
        self.values[r * self.cols + c] = v


class BiasGene(Gene):
    def __init__(self, size: int, init_scale: float = 0.2):
        self.size = size
        self.values = [random.uniform(-init_scale, init_scale) for _ in range(size)]


class Genome:
    """
    Genome encoding a simple fully-connected MLP: input -> hidden... -> output
    - Layer sizes (hidden layers) are mutable.
    - Weights and biases are mutable.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers: Optional[List[int]] = None):
        if hidden_layers is None:
            hidden_layers = [max(4, input_size // 2)]
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = list(hidden_layers)

        self.layer_genes: List[LayerGene] = []
        self.weight_genes: List[WeightsGene] = []
        self.bias_genes: List[BiasGene] = []

        sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(sizes) - 1):
            self.layer_genes.append(LayerGene(sizes[i], sizes[i + 1]))
            self.weight_genes.append(WeightsGene(rows=sizes[i + 1], cols=sizes[i]))
            self.bias_genes.append(BiasGene(size=sizes[i + 1]))

    def clone(self) -> 'Genome':
        g = Genome(self.input_size, self.output_size, list(self.hidden_layers))
        for i in range(len(self.weight_genes)):
            g.weight_genes[i].values = list(self.weight_genes[i].values)
            g.bias_genes[i].values = list(self.bias_genes[i].values)
        return g

    # Genetic operators
    def mutate(self,
               weight_sigma: float = 0.1,
               bias_sigma: float = 0.05,
               arch_mutate_prob: float = 0.1,
               max_hidden_size: int = 64):
        # Mutate weights
        for wg in self.weight_genes:
            for i in range(len(wg.values)):
                if random.random() < 0.2:
                    wg.values[i] += random.gauss(0.0, weight_sigma)

        # Mutate biases
        for bg in self.bias_genes:
            for i in range(len(bg.values)):
                if random.random() < 0.2:
                    bg.values[i] += random.gauss(0.0, bias_sigma)

        # Occasionally mutate architecture: add/remove a hidden layer or change its size
        if random.random() < arch_mutate_prob:
            choice = random.choice(["add", "remove", "resize"]) if self.hidden_layers else "add"
            if choice == "add" and len(self.hidden_layers) < 3:
                pos = random.randint(0, len(self.hidden_layers))
                new_size = random.randint(4, min(max_hidden_size, max(8, self.input_size)))
                self.hidden_layers.insert(pos, new_size)
                self._rebuild_params()
            elif choice == "remove" and len(self.hidden_layers) > 1:
                pos = random.randrange(len(self.hidden_layers))
                del self.hidden_layers[pos]
                self._rebuild_params()
            elif choice == "resize" and self.hidden_layers:
                pos = random.randrange(len(self.hidden_layers))
                self.hidden_layers[pos] = max(4, min(max_hidden_size, self.hidden_layers[pos] + random.randint(-4, 4)))
                self._rebuild_params()

    def crossover(self, other: 'Genome') -> 'Genome':
        # One-point crossover on hidden layers shape, then blend weights/biases
        cut = random.randint(0, min(len(self.hidden_layers), len(other.hidden_layers)))
        new_hidden = self.hidden_layers[:cut] + other.hidden_layers[cut:]
        child = Genome(self.input_size, self.output_size, new_hidden)
        for i in range(len(child.weight_genes)):
            a = self.weight_genes[i].values if i < len(self.weight_genes) else child.weight_genes[i].values
            b = other.weight_genes[i].values if i < len(other.weight_genes) else child.weight_genes[i].values
            alpha = 0.5
            child.weight_genes[i].values = [alpha * av + (1 - alpha) * bv for av, bv in zip(a, b)]

            a_b = self.bias_genes[i].values if i < len(self.bias_genes) else child.bias_genes[i].values
            b_b = other.bias_genes[i].values if i < len(other.bias_genes) else child.bias_genes[i].values
            child.bias_genes[i].values = [alpha * av + (1 - alpha) * bv for av, bv in zip(a_b, b_b)]
        return child

    def _rebuild_params(self):
        # Recreate parameter genes to match updated architecture; keep old values where possible
        sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        new_w: List[WeightsGene] = []
        new_b: List[BiasGene] = []
        for i in range(len(sizes) - 1):
            rows, cols = sizes[i + 1], sizes[i]
            wg = WeightsGene(rows, cols)
            bg = BiasGene(rows)
            # Try to copy overlapping block from previous params
            if i < len(self.weight_genes):
                old = self.weight_genes[i]
                old_rows = min(old.rows, rows)
                old_cols = min(old.cols, cols)
                for r in range(old_rows):
                    for c in range(old_cols):
                        wg.set(r, c, old.get(r, c))
            if i < len(self.bias_genes):
                oldb = self.bias_genes[i]
                old_len = min(len(oldb.values), rows)
                for j in range(old_len):
                    bg.values[j] = oldb.values[j]
            new_w.append(wg)
            new_b.append(bg)
        self.weight_genes = new_w
        self.bias_genes = new_b


class EvoMLP:
    def __init__(self, genome: Genome):
        self.g = genome

    @staticmethod
    def _tanh(v: float) -> float:
        return math.tanh(v)

    def forward(self, x: List[float]) -> List[float]:
        a = x
        for i in range(len(self.g.weight_genes)):
            W = self.g.weight_genes[i]
            b = self.g.bias_genes[i]
            out = [0.0] * W.rows
            for r in range(W.rows):
                s = b.values[r]
                # Dot product
                # Unrolled indexing for speed clarity
                for c in range(W.cols):
                    s += W.get(r, c) * (a[c] if c < len(a) else 0.0)
                # Activation: tanh for hidden, tanh for output to get [-1, 1]
                out[r] = self._tanh(s)
            a = out
        return a

    def forward_layers(self, x: List[float]) -> List[List[float]]:
        """
        Compute per-layer activations including the input layer.

        Returns a list of layers: [input, layer1, layer2, ..., output]
        Each layer is a list of floats.
        """
        layers: List[List[float]] = [list(x)]
        a = x
        for i in range(len(self.g.weight_genes)):
            W = self.g.weight_genes[i]
            b = self.g.bias_genes[i]
            out = [0.0] * W.rows
            for r in range(W.rows):
                s = b.values[r]
                for c in range(W.cols):
                    s += W.get(r, c) * (a[c] if c < len(a) else 0.0)
                out[r] = self._tanh(s)
            layers.append(out)
            a = out
        return layers


class EvoNeuralController(NeuralNetworkInterface):
    """
    Neural controller that consumes muscle stretch ratios and outputs activations in [-1, 1].

    Input vector: [stretch_0, stretch_1, ..., stretch_{N-1}], where stretch is
    current_length / rest_length. Values typically around ~1.0; we center them by subtracting 1.0.
    """

    def __init__(self, num_muscles: int, genome: Optional[Genome] = None):
        super().__init__(num_muscles)
        if genome is None:
            genome = Genome(input_size=num_muscles, output_size=num_muscles, hidden_layers=[max(4, num_muscles // 2)])
        self.genome = genome
        self.mlp = EvoMLP(self.genome)

    def get_activations(self, state: dict) -> list:
        stretches = [1.0] * self.num_muscles
        muscles = state.get('muscles', [])
        for i in range(min(len(muscles), self.num_muscles)):
            st = muscles[i].get('stretch', 1.0)
            # Center around 0 and clamp to a reasonable range
            stretches[i] = max(-1.0, min(1.0, st - 1.0))
        outputs = self.mlp.forward(stretches)
        # Already in [-1, 1] because of tanh
        return [max(-1.0, min(1.0, o)) for o in outputs]


def fitness_distance_speed(distance_traveled: float,
                           max_speed: float,
                           distance_weight: float = 0.75,
                           speed_weight: float = 0.25) -> float:
    """
    Compute fitness as a weighted combination of distance traveled and maximum speed.

    By default, prioritizes distance (0.75) and also rewards max speed (0.25).

    Args:
        distance_traveled: Total or horizontal distance the cat moved.
        max_speed: Maximum speed achieved during the episode.
        distance_weight: Weight for distance component (default 0.75).
        speed_weight: Weight for max-speed component (default 0.25).

    Returns:
        A scalar fitness value.
    """
    return distance_weight * float(distance_traveled) + speed_weight * float(max_speed)


class EvolutionaryAlgorithm:
    """
    Simple GA to evolve genomes for the controller.
    Usage:
        ea = EvolutionaryAlgorithm(input_size, output_size, pop_size=20)
        def eval_fn(genome):
            # Return a fitness score (higher is better)
            ...
        ea.evolve(eval_fn, generations=10)
        best = ea.get_best()
        controller = EvoNeuralController(num_muscles=output_size, genome=best)
    """

    def __init__(self, input_size: int, output_size: int, pop_size: int = 20):
        self.input_size = input_size
        self.output_size = output_size
        self.pop_size = pop_size
        self.population: List[Genome] = [Genome(input_size, output_size) for _ in range(pop_size)]
        self.fitness: List[float] = [float('-inf')] * pop_size

    def evaluate(self, evaluator: Callable[[Genome], float]):
        for i, g in enumerate(self.population):
            self.fitness[i] = evaluator(g)

    def select_parents(self) -> List[int]:
        # Tournament selection
        idxs = list(range(self.pop_size))
        selected = []
        for _ in range(self.pop_size):
            a, b = random.sample(idxs, 2)
            selected.append(a if self.fitness[a] >= self.fitness[b] else b)
        return selected

    def reproduce(self, parent_indices: List[int]) -> List[Genome]:
        new_pop: List[Genome] = []
        for i in range(0, self.pop_size, 2):
            p1 = self.population[parent_indices[i]]
            p2 = self.population[parent_indices[i + 1]] if i + 1 < self.pop_size else self.population[parent_indices[0]]
            child1 = p1.crossover(p2)
            child2 = p2.crossover(p1)
            child1.mutate()
            child2.mutate()
            new_pop.extend([child1, child2])
        return new_pop[:self.pop_size]

    def _save_best(self, generation: int, directory: str = "checkpoints"):
        os.makedirs(directory, exist_ok=True)
        best = self.get_best()
        # Save as pickle for simplicity
        path = os.path.join(directory, f"gen_{generation:04d}_best.pkl")
        print(f"Saving best genome to {path}")
        with open(path, 'wb') as f:
            pickle.dump(best, f)

    def evolve(self, evaluator: Callable[[Genome], float], generations: int = 10):
        for gen in range(1, generations + 1):
            self.evaluate(evaluator)
            # Save best every 20 generations
            if gen % 3 == 0:
                self._save_best(gen)
            parents = self.select_parents()
            self.population = self.reproduce(parents)

    def get_best(self) -> Genome:
        if not any(f != float('-inf') for f in self.fitness):
            # If not evaluated yet, return a random
            return random.choice(self.population).clone()
        best_idx = max(range(self.pop_size), key=lambda i: self.fitness[i])
        return self.population[best_idx].clone()
