from cat import NeuralNetworkInterface
import os


class LegExtensionTestController(NeuralNetworkInterface):
    """
    Simple, simulator-friendly controller for muscle testing.

    Notes
    - Accepts an optional `genome` argument so it can be used where an
      evolutionary controller would normally be injected by the simulator.
      The genome is ignored; this controller is purely rule-based.
    - Supports several fixed activation patterns to help debug/test muscles.

    How to choose a pattern
    - Pass parameters to the constructor, OR
    - Set environment variables before launching the simulator:
        TEST_MODE   = zeros | ones | alternate | index | hips | knees
        TEST_INDEX  = <int>   (used by mode=index)
        TEST_VALUE  = <float> (default 1.0)
    """

    def __init__(self, num_muscles: int, genome=None,
                 mode: str | None = None,
                 active_index: int | None = None,
                 value: float | None = None):
        super().__init__(num_muscles)
        # Keep genome attribute for compatibility with simulator expectations
        self.genome = genome

        # Resolve configuration from args or environment variables
        self.mode = (mode or os.getenv('TEST_MODE', 'zeros')).strip().lower()
        try:
            self.active_index = int(os.getenv('TEST_INDEX', str(active_index if active_index is not None else -1)))
        except Exception:
            self.active_index = -1
        try:
            default_val = value if value is not None else 1.0
            self.value = float(os.getenv('TEST_VALUE', str(default_val)))
        except Exception:
            self.value = 1.0

        # Clamp value into valid range [-1.0, 1.0] (muscles support negative too)
        self.value = max(-1.0, min(1.0, self.value))

    def get_activations(self, state: dict) -> list:
        """Return activation pattern according to selected testing mode.

        Args:
            state: Current state of the cat (unused for fixed patterns)

        Returns:
            List of activations in [-1.0, 1.0] for each muscle
        """
        n = self.num_muscles
        v = self.value
        mode = self.mode

        if mode in ('zeros', 'off', 'all_off', 'extend'):
            # Deactivate all muscles – legs will extend under gravity/springs
            return [0.0] * n

        if mode in ('ones', 'on', 'all_on', 'contract'):
            # Full contraction everywhere
            return [v] * n

        if mode in ('alternate', 'alt'):
            # Even indices on, odd off
            return [v if (i % 2 == 0) else 0.0 for i in range(n)]

        if mode in ('index', 'single', 'one'):
            # Only a single muscle is activated
            idx = self.active_index if (0 <= self.active_index < n) else 0
            out = [0.0] * n
            out[idx] = v
            return out

        if mode in ('hips', 'hip_only'):
            # Assuming muscles are ordered [hip, knee] per leg
            return [v if (i % 2 == 0) else 0.0 for i in range(n)]

        if mode in ('knees', 'knee_only'):
            # Assuming muscles are ordered [hip, knee] per leg
            return [v if (i % 2 == 1) else 0.0 for i in range(n)]

        # Fallback: safe default
        return [1.0] * n