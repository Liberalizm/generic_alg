import math
from typing import List, Dict

import Box2D


class CatStats:
    """Tracks distance and speed statistics for a cat."""

    def __init__(self, spawn_x: float):
        self.spawn_x = spawn_x
        self.spawn_y = 0.0  # Will be set when cat is created
        self.max_speed = 0.0
        self.total_distance = 0.0
        self.last_position = None
        self.speed_history = []
        self.history_max_size = 60  # ~1 second at 60 FPS for average

    def update(self, current_x: float, current_y: float, velocity_x: float, velocity_y: float, dt: float):
        """Update statistics based on current position and velocity."""
        # Calculate current speed (magnitude of velocity)
        current_speed = math.sqrt(velocity_x ** 2 + velocity_y ** 2)

        # Update max speed
        if current_speed > self.max_speed:
            self.max_speed = current_speed

        # Track speed for average calculation
        self.speed_history.append(current_speed)
        if len(self.speed_history) > self.history_max_size:
            self.speed_history.pop(0)

        # Update total distance traveled
        if self.last_position is not None:
            dx = current_x - self.last_position[0]
            dy = current_y - self.last_position[1]
            self.total_distance += math.sqrt(dx ** 2 + dy ** 2)

        self.last_position = (current_x, current_y)

    def get_distance_from_spawn(self, current_x: float) -> float:
        """Get horizontal distance from spawn point."""
        return current_x - self.spawn_x

    def get_current_speed(self) -> float:
        """Get the most recent speed."""
        if self.speed_history:
            return self.speed_history[-1]
        return 0.0

    def get_average_speed(self) -> float:
        """Get average speed over recent history."""
        if self.speed_history:
            return sum(self.speed_history) / len(self.speed_history)
        return 0.0

    def get_max_speed(self) -> float:
        """Get maximum speed achieved."""
        return self.max_speed

    def get_joint_angles(self, legs: list) -> dict:
        """
        Get angles of all joints from the cat's legs.

        Args:
            legs: List of leg dictionaries containing hip and knee joints

        Returns:
            Dictionary with joint angles in radians:
            - 'hip_angles': list of hip joint angles (4 values)
            - 'knee_angles': list of knee joint angles (4 values)
            - 'all_angles': flat list of all angles [hip0, knee0, hip1, knee1, ...]
        """
        hip_angles = []
        knee_angles = []
        all_angles = []

        for leg in legs:
            hip_angle = leg['hip'].angle
            knee_angle = leg['knee'].angle

            hip_angles.append(hip_angle)
            knee_angles.append(knee_angle)
            all_angles.extend([hip_angle, knee_angle])

        return {
            'hip_angles': hip_angles,
            'knee_angles': knee_angles,
            'all_angles': all_angles,
        }


class NeuralNetworkInterface:
    """
    Interface for connecting muscles to a neural network.
    Override get_activations() to implement your neural network.
    """

    def __init__(self, num_muscles: int):
        self.num_muscles = num_muscles

    def get_activations(self, state: dict) -> list:
        """
        Get activation values for all muscles from the neural network.

        Args:
            state: Dictionary containing the current state of the cat
                   (positions, velocities, angles, etc.)

        Returns:
            List of activation values (-1.0 to 1.0) for each muscle
        """
        # Placeholder - return zeros (no activation)
        # Override this method to implement your neural network
        return [0.0] * self.num_muscles

    def get_state(self, cat: 'Cat') -> dict:
        """Extract state information from the cat for neural network input."""
        state = {
            'torso_position': (cat.body.position.x, cat.body.position.y),
            'torso_angle': cat.body.angle,
            'torso_velocity': (cat.body.linearVelocity.x, cat.body.linearVelocity.y),
            'torso_angular_velocity': cat.body.angularVelocity,
            'legs': [],
            'muscles': [],
        }

        for leg in cat.legs:
            leg_state = {
                'hip_angle': leg['hip'].angle,
                'hip_speed': leg['hip'].speed,
                'knee_angle': leg['knee'].angle,
                'knee_speed': leg['knee'].speed,
            }
            state['legs'].append(leg_state)

        for muscle in cat.muscles:
            muscle_state = {
                'energy_percent': muscle.get_energy_percent(),
                'activation': muscle.activation,
                # Provide stretch ratio for controllers that rely on muscle length feedback
                # stretch ~ 1.0 at rest; >1.0 extended; <1.0 compressed
                'stretch': muscle.get_stretch_ratio(),
            }
            state['muscles'].append(muscle_state)

        return state


class Muscle:
    """
    A muscle that applies forces between two attachment points on Box2D bodies.

    Features:
    - Energy pool that depletes based on work done (force * distance or load)
    - Energy regenerates at a constant rate
    - Maximum force scales linearly with remaining energy percentage
    - Spring-like behavior to maintain rest length
    - Negative activation when muscle is extended beyond rest length
    """

    def __init__(
            self,
            body_a: Box2D.b2Body,
            body_b: Box2D.b2Body,
            local_anchor_a: tuple,
            local_anchor_b: tuple,
            max_force: float = 100.0,
            max_energy: float = 100.0,
            energy_regen_rate: float = 10.0,  # energy units per second
            energy_cost_factor: float = 0.1,  # energy cost per unit of work
            rest_length: float = None,  # Rest length for spring behavior
            spring_stiffness: float = 50.0,  # Spring constant (N/m)
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.local_anchor_a = local_anchor_a
        self.local_anchor_b = local_anchor_b

        # Force parameters
        self.max_force = max_force

        # Energy parameters
        self.max_energy = max_energy
        self.energy = max_energy  # Start fully charged
        self.energy_regen_rate = energy_regen_rate
        self.energy_cost_factor = energy_cost_factor

        # Spring parameters
        self.rest_length = rest_length  # Will be set based on rest position
        self.spring_stiffness = spring_stiffness

        # Current activation from neural network (-1.0 to 1.0)
        # Positive: contracting, Negative: extended (resisting)
        self.activation = 0.0

        # Telemetry for UI/debugging
        self.last_force = 0.0   # absolute force magnitude along line of action (N)
        self.last_power = 0.0   # non-negative mechanical power from activation component (W, arbitrary units)

    def get_world_anchors(self):
        """Get world coordinates of muscle attachment points."""
        anchor_a = self.body_a.GetWorldPoint(self.local_anchor_a)
        anchor_b = self.body_b.GetWorldPoint(self.local_anchor_b)
        return anchor_a, anchor_b

    def get_energy_percent(self) -> float:
        """Returns the remaining energy as a percentage (0.0 to 1.0)."""
        return self.energy / self.max_energy

    def get_effective_max_force(self) -> float:
        """Maximum force scales linearly with remaining energy percentage."""
        return self.max_force * self.get_energy_percent()

    def get_current_length(self) -> float:
        """Get the current distance between attachment points."""
        anchor_a, anchor_b = self.get_world_anchors()
        dx = anchor_b.x - anchor_a.x
        dy = anchor_b.y - anchor_a.y
        return math.sqrt(dx * dx + dy * dy)

    def get_stretch_ratio(self) -> float:
        """
        Returns the stretch ratio relative to rest length.
        < 1.0: muscle is compressed (shorter than rest)
        = 1.0: muscle is at rest length
        > 1.0: muscle is extended (longer than rest)
        """
        if self.rest_length is None or self.rest_length < 0.001:
            return 1.0
        return self.get_current_length() / self.rest_length

    def apply_force(self, activation: float, dt: float):
        """
        Apply muscle force based on activation level (from neural network).
        Now supports spring behavior with automatic rest position maintenance.

        Args:
            activation: Value between -1.0 and 1.0
                       Positive: contract (pull bodies together)
                       Negative: API output when muscle is extended beyond rest
            dt: Time step in seconds
        """
        self.activation = max(-1.0, min(1.0, activation))

        # Get attachment points in world coordinates
        anchor_a, anchor_b = self.get_world_anchors()

        # Direction from A to B
        dx = anchor_b.x - anchor_a.x
        dy = anchor_b.y - anchor_a.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 0.001:
            return

        # Normalize direction
        dir_x = dx / distance
        dir_y = dy / distance

        # Calculate effective force components
        effective_max_force = self.get_effective_max_force()
        activation_force = effective_max_force * self.activation

        # Spring force pulls back toward rest length (if defined)
        if self.rest_length is not None:
            spring_force = (distance - self.rest_length) * self.spring_stiffness
        else:
            spring_force = 0.0

        # Total force magnitude along the line A->B
        force_magnitude = activation_force + spring_force

        # print(f"Muscle apply_force={force_magnitude} activation={self.activation:.2f}, distance={distance:.2f}, effective_max_force={effective_max_force:.2f}, rest_length={self.rest_length:.2f}, spring_stiffness={self.spring_stiffness:.2f}")
        # Determine force direction based on activation sign
        # Positive activation: contract (pull together)
        # Negative activation: extend (push apart) - resisting compression
        # Contract: force pulls bodies together
        force_x = dir_x * force_magnitude
        force_y = dir_y * force_magnitude
        self.body_a.ApplyForce((force_x, force_y), anchor_a, wake=True)
        self.body_b.ApplyForce((-force_x, -force_y), anchor_b, wake=True)

        # Calculate energy cost based on mechanical work from the activation component only
        # Power ~ |F_activation| * |relative_speed_along_line|
        va = self.body_a.GetLinearVelocityFromWorldPoint(anchor_a)
        vb = self.body_b.GetLinearVelocityFromWorldPoint(anchor_b)
        rel_vx = vb.x - va.x
        rel_vy = vb.y - va.y
        rel_speed_along = abs(rel_vx * dir_x + rel_vy * dir_y)

        power_activation = abs(activation_force) * rel_speed_along
        # Telemetry outputs
        self.last_force = abs(force_magnitude)
        self.last_power = power_activation
        energy_cost = power_activation * self.energy_cost_factor * dt
        # Clamp energy within [0, max_energy]
        self.energy = max(0.0, min(self.max_energy, self.energy - energy_cost))

    def regenerate_energy(self, dt: float):
        """Regenerate energy at a constant rate."""
        self.energy = min(self.max_energy, self.energy + self.energy_regen_rate * dt)

    def update(self, activation: float, dt: float):
        """
        Main update method to be called each simulation step.

        Args:
            activation: Neural network output (-1.0 to 1.0)
            dt: Time step in seconds
        """
        #print("Update fun is triggered")
        self.apply_force(activation, dt)
        self.regenerate_energy(dt)


class Cat:
    """
    A simple articulated 'cat' built with Box2D:
    - 1 torso (single dynamic body)
    - 4 legs, each with 2 segments (upper and lower)
    - Each leg has 2 revolute joints (hip and knee)
    - Each joint is spanned by 2 antagonistic 'muscles' implemented as distance joints

    You can call update(t) every step to animate the muscles.
    """

    def __init__(self, world: Box2D.b2World, position=(10, 15), cat_index: int = 0):
        self.world = world

        self.scale = 9.5

        # Collision filtering: use group index for filtering instead of category bits
        # Negative group index means bodies with same index don't collide with each other
        # All cats use category 0x0002 and only collide with ground (0x0001)
        self.category_bits = 0x0002  # All cats same category
        self.mask_bits = 0x0001      # Only collide with ground
        self.group_index = -(cat_index + 1)  # Negative = don't collide with same group

        # Torso
        self.body = world.CreateDynamicBody(position=position, angle=0)
        # Torso roughly 2.0m wide x 0.6m tall
        self.body.CreatePolygonFixture(
            box=(1.0 * self.scale, 0.3 * self.scale),
            density=1.0,
            friction=0.4,
            categoryBits=self.category_bits,
            maskBits=self.mask_bits,
            groupIndex=self.group_index
        )
        # Tag for rendering
        self.body.userData = {"entity": "cat", "part": "torso"}

        # Initialize stats tracking
        self.stats = CatStats(spawn_x=position[0])
        self.stats.spawn_y = position[1]

        # Geometry parameters for legs
        self.hip_offset_x = 0.8 * self.scale  # hips placed forward/back on the torso
        self.hip_offset_y = -0.25 * self.scale
        self.upper_len = 0.6 * self.scale
        self.lower_len = 0.6 * self.scale
        self.leg_thickness = 0.1 * self.scale

        # Storage
        self.legs: List[Dict] = []
        self.muscles: List[Muscle] = []

        self.nn_interface: NeuralNetworkInterface = None

        # Create 4 legs: front-left, front-right, back-left, back-right
        x_offsets = [-self.hip_offset_x, self.hip_offset_x]  # back, front
        side_signs = [-1, 1]  # left, right (for small lateral spread)

        for x_off in x_offsets:
            for side in side_signs:
                is_front = x_off > 0
                leg = self._create_leg(
                    anchor_local=(x_off, self.hip_offset_y),
                    lateral=0.05 * self.scale * side,
                    is_front=is_front,
                )
                self.legs.append(leg)

    def calculate_rest_muscle_lengths(self, hip_angle_deg: float = 45.0, knee_angle_deg: float = 90.0):
        """
        Calculate and set rest lengths for all muscles based on target joint angles.

        Args:
            hip_angle_deg: Hip angle in degrees (45° = leg angled forward/back from vertical)
            knee_angle_deg: Knee angle in degrees (90° = right angle at knee)
        """
        hip_angle_rad = math.radians(hip_angle_deg)
        knee_angle_rad = math.radians(knee_angle_deg)

        for i, leg in enumerate(self.legs):
            muscle_idx = i * 2
            is_front = (i >= 2)  # First 2 legs are back, last 2 are front

            if is_front:
                x_off = self.leg_thickness * 0.75
            else:
                x_off = -self.leg_thickness * 0.75

            # Hip muscle rest length calculation
            # Hip muscle connects torso center (0,0) to point on upper leg
            if muscle_idx < len(self.muscles):
                hip_muscle = self.muscles[muscle_idx]

                # Upper leg rotated by hip_angle from vertical
                # Local anchor on upper leg: (-x_off, -upper_len * 0.3)
                # When hip is at rest angle, calculate where this point would be
                # relative to torso center

                # Hip joint is at (hip_offset_x, hip_offset_y) on torso
                # Upper leg hangs from hip, rotated by hip_angle
                hip_local = (self.hip_offset_x if is_front else -self.hip_offset_x, self.hip_offset_y)

                # Point on upper leg in upper leg's local coords
                upper_anchor_local = (-x_off, -self.upper_len * 0.3)

                # Rotate upper leg anchor by hip angle (relative to vertical)
                cos_h = math.cos(-hip_angle_rad)
                sin_h = math.sin(-hip_angle_rad)
                upper_anchor_rotated = (
                    upper_anchor_local[0] * cos_h - upper_anchor_local[1] * sin_h,
                    upper_anchor_local[0] * sin_h + upper_anchor_local[1] * cos_h
                )

                # Position relative to torso center
                upper_anchor_world = (
                    hip_local[0] + upper_anchor_rotated[0],
                    hip_local[1] - self.upper_len / 2.0 + upper_anchor_rotated[1]
                )

                # Distance from torso center (0,0) to this point
                hip_muscle.rest_length = math.sqrt(
                    upper_anchor_world[0] ** 2 + upper_anchor_world[1] ** 2
                )

            # Knee muscle rest length calculation
            # Knee muscle connects upper leg to lower leg across the knee joint
            if muscle_idx + 1 < len(self.muscles):
                knee_muscle = self.muscles[muscle_idx + 1]

                # Upper leg anchor: (x_off, upper_len * 0.1) - near bottom of upper leg
                # Lower leg anchor: (x_off, -lower_len * 0.1) - near top of lower leg

                upper_anchor = (x_off, self.upper_len * 0.1)  # Below center of upper leg
                lower_anchor = (x_off, -self.lower_len * 0.1)  # Above center of lower leg

                # At 90° knee angle, the lower leg is perpendicular to upper leg
                # The muscle spans across this angle
                # Distance calculation considering the knee bend

                # Vector from knee joint to upper anchor (in upper leg frame)
                upper_to_knee = (upper_anchor[0], upper_anchor[1] + self.upper_len / 2.0)

                # Vector from knee joint to lower anchor (in lower leg frame, then rotated)
                lower_from_knee = (lower_anchor[0], lower_anchor[1] - self.lower_len / 2.0)

                # Rotate lower leg vector by knee angle
                cos_k = math.cos(math.pi - knee_angle_rad)
                sin_k = math.sin(math.pi - knee_angle_rad)
                lower_rotated = (
                    lower_from_knee[0] * cos_k - lower_from_knee[1] * sin_k,
                    lower_from_knee[0] * sin_k + lower_from_knee[1] * cos_k
                )

                # Distance between the two anchor points
                dx = lower_rotated[0] - upper_to_knee[0]
                dy = lower_rotated[1] - upper_to_knee[1]
                knee_muscle.rest_length = math.sqrt(dx * dx + dy * dy)

    def set_to_rest_position(self, hip_angle_deg: float = 45.0, knee_angle_deg: float = 90.0):
        """
        Move the cat to the rest position and calculate muscle rest lengths.
        Call this after creating the cat to establish the rest state.
        """
        # First, set joint motors to move to target angles
        # Then calculate rest lengths
        self.calculate_rest_muscle_lengths(hip_angle_deg, knee_angle_deg)

    def _create_leg(self, anchor_local: tuple, lateral: float, is_front: bool) -> Dict:
        # ... existing bone and joint creation code stays the same ...
        # Мировая позиция тазобедренного сустава
        hip_world = self.body.GetWorldPoint((anchor_local[0], anchor_local[1]))

        # Upper bone
        upper = self.world.CreateDynamicBody(
            position=(hip_world.x + lateral, hip_world.y - self.upper_len / 2.0)
        )
        upper.CreatePolygonFixture(
            box=(self.leg_thickness, self.upper_len / 2.0),
            density=0.5,
            friction=0.8,
            categoryBits=self.category_bits,
            maskBits=self.mask_bits,
            groupIndex=self.group_index
        )
        upper.userData = {"entity": "cat", "part": "upper_leg"}

        # Lower bone
        knee_world = (upper.position.x, upper.position.y - self.upper_len / 2.0)
        lower = self.world.CreateDynamicBody(
            position=(knee_world[0], knee_world[1] - self.lower_len / 2.0)
        )
        lower.CreatePolygonFixture(
            box=(self.leg_thickness, self.lower_len / 2.0),
            density=0.8,
            friction=0.9,
            categoryBits=self.category_bits,
            maskBits=self.mask_bits,
            groupIndex=self.group_index
        )
        lower.userData = {"entity": "cat", "part": "lower_leg"}

        # Joints
        hip = self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=upper,
            localAnchorA=anchor_local,
            localAnchorB=(-lateral, self.upper_len / 2.0),
            enableLimit=True,
            lowerAngle=math.radians(-75) if is_front else math.radians(-85),
            upperAngle=math.radians(85) if is_front else math.radians(75),
            enableMotor=False,
        )
        knee = self.world.CreateRevoluteJoint(
            bodyA=upper,
            bodyB=lower,
            localAnchorA=(0.0, -self.upper_len / 2.0),
            localAnchorB=(0.0, self.lower_len / 2.0),
            enableLimit=True,
            lowerAngle=math.radians(5) if is_front else math.radians(-170),
            upperAngle=math.radians(170) if is_front else math.radians(-5),
            enableMotor=False,
        )

        # Create force-based muscles
        # Hip muscle attaches from torso center to upper leg
        # Knee muscle attaches from upper leg to lower leg
        # Use consistent attachment points for all legs

        if is_front:
            x_off = self.leg_thickness * 0.75
        else:
            x_off = -self.leg_thickness * 0.75

        # Hip muscle (torso to upper leg) - attaches to front of upper leg
        hip_muscle = Muscle(
            body_a=self.body,
            body_b=upper,
            local_anchor_a=(0, 0),  # Point on torso near this leg's hip
            local_anchor_b=(-x_off, -self.upper_len * 0.3),  # Lower part of upper leg
            max_force=480.0 * self.scale,
            max_energy=10000.0,
            energy_regen_rate=20.0,
            energy_cost_factor=0.03,
            rest_length=(self.upper_len * 0.1 + self.lower_len * 0.1) * 5.5,
            spring_stiffness=100.0 * self.scale,  # Strong spring to hold position
        )
        self.muscles.append(hip_muscle)


        # Knee muscle (upper leg to lower leg) - spans the knee joint
        knee_muscle = Muscle(
            body_a=upper,
            body_b=lower,
            local_anchor_a=(x_off, self.upper_len * 0.1),  # Lower part of upper leg
            local_anchor_b=(x_off, -self.lower_len * 0.1),   # Upper part of lower leg
            max_force=360.0 * self.scale,
            max_energy=10000.0,
            energy_regen_rate=20.0,
            energy_cost_factor=0.03,
            rest_length= (self.upper_len * 0.1 + self.lower_len * 0.1) * 4,
            spring_stiffness=80.0 * self.scale,  # Strong spring to hold position
        )
        self.muscles.append(knee_muscle)

        return {
            "upper": upper,
            "lower": lower,
            "hip": hip,
            "knee": knee,
            "hip_muscle": hip_muscle,
            "knee_muscle": knee_muscle,
        }

    def set_neural_network(self, nn_interface: NeuralNetworkInterface):
        """Set the neural network interface for muscle control."""
        self.nn_interface = nn_interface

    def update(self, dt: float):
        """
        Update muscles based on neural network output.

        Args:
            dt: Time step in seconds
        """
        # Update stats
        pos = self.body.position
        vel = self.body.linearVelocity
        self.stats.update(pos.x, pos.y, vel.x, vel.y, dt)

        if self.nn_interface:
            state = self.nn_interface.get_state(self)
            activations = self.nn_interface.get_activations(state)

            for i, muscle in enumerate(self.muscles):
                activation = activations[i] if i < len(activations) else 0.0
                muscle.update(activation, dt)
        else:
            # Fallback: no activation
            for muscle in self.muscles:
                muscle.update(0.0, dt)

    def get_position(self):
        return self.body.position

    def get_angle(self):
        return self.body.angle

    def destroy(self):
        """Safely destroy all Box2D objects belonging to this cat.

        Box2D will normally destroy joints when their bodies are destroyed,
        but we explicitly destroy joints first for clarity and to avoid any
        potential dangling references in our own structures.
        """
        try:
            # Destroy joints first
            for leg in getattr(self, 'legs', []) or []:
                hip = leg.get('hip')
                knee = leg.get('knee')
                if hip is not None:
                    try:
                        self.world.DestroyJoint(hip)
                    except Exception:
                        pass
                if knee is not None:
                    try:
                        self.world.DestroyJoint(knee)
                    except Exception:
                        pass

            # Destroy leg bodies
            for leg in getattr(self, 'legs', []) or []:
                upper = leg.get('upper')
                lower = leg.get('lower')
                if upper is not None:
                    try:
                        self.world.DestroyBody(upper)
                    except Exception:
                        pass
                if lower is not None:
                    try:
                        self.world.DestroyBody(lower)
                    except Exception:
                        pass

            # Destroy torso/body last
            if getattr(self, 'body', None) is not None:
                try:
                    self.world.DestroyBody(self.body)
                except Exception:
                    pass
        finally:
            # Clear references to help GC and avoid re-use
            self.legs = []
            self.muscles = []
            self.nn_interface = None