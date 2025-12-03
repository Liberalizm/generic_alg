from cat import NeuralNetworkInterface


class LegExtensionTestController(NeuralNetworkInterface):
    """
    Test controller that extends legs to maximum by deactivating all muscles.

    Since muscles can only contract (pull), setting activation to 0.0
    allows gravity and joint mechanics to extend the legs to their limits.
    """

    def __init__(self, num_muscles: int):
        super().__init__(num_muscles)

    def get_activations(self, state: dict) -> list:
        """
        Return zero activation for all muscles to allow maximum leg extension.

        Args:
            state: Current state of the cat (unused in this simple controller)

        Returns:
            List of 0.0 activations for all muscles
        """
        # Deactivate all muscles - legs will extend under gravity
        return [-1.0] * self.num_muscles