from cat import NeuralNetworkInterface
import random
import math

class SimpleNeuralNetwork(NeuralNetworkInterface):
    """Simple neural network for muscle control with random weights."""

    def __init__(self, num_muscles: int):
        super().__init__(num_muscles)
        # Random weights for demonstration
        self.weights = [[random.uniform(-1, 1) for _ in range(12)] for _ in range(num_muscles)]
        self.biases = [random.uniform(-0.5, 0.5) for _ in range(num_muscles)]
        self.phase = random.uniform(0, 2 * math.pi)
        self.frequency = random.uniform(1.0, 3.0)
        self.time = 0.0

    def get_activations(self, state: dict) -> list:
        """Generate muscle activations based on state."""
        self.time += 1/60  # Approximate dt

        activations = []
        for i in range(self.num_muscles):
            # Simple oscillating pattern with state influence
            base = math.sin(self.time * self.frequency * 2 * math.pi + self.phase + i * 0.5)
            activation = (base + 1) / 2  # Normalize to 0-1

            # Add some variation based on leg angles
            if state['legs'] and i // 2 < len(state['legs']):
                leg = state['legs'][i // 2]
                if i % 2 == 0:  # Hip muscle
                    activation += leg['hip_angle'] * 0.1
                else:  # Knee muscle
                    activation += leg['knee_angle'] * 0.1

            activation = max(0.0, min(1.0, activation + self.biases[i] * 0.2))
            activations.append(activation)

        return activations