"""An actor is a class which has some sort of system model, a specified control policy,
and a set of sensors. It is embedded in some environment.

Typical Flow:
    1. The actor senses the environment and model state. (Assume direct observability)
    2. The actor applies the control policy.
    3. The model evolves with time according to the policy and environment.
    4. Typically the environment does not change as a result.
    4. Repeat.
"""

import jax.numpy as jnp
from pydrostat.model.dynamics import DynamicModel
from pydrostat.control.policy_interface import IPolicy


class Actor:
    def __init__(
        self,
        model: DynamicModel,
        policy: IPolicy,
        sensors=None,
        initial_state: jnp.ndarray = None,
    ):
        """Initialize the actor with a model, policy, and optional sensors."""
        self.model = model
        self.policy = policy
        self.sensors = sensors if sensors is not None else []
        self.state = (
            initial_state if initial_state is not None else jnp.zeros(model.num_states)
        )
        self.control = jnp.zeros(model.num_controls)

        self.current_obstacles = (
            []
        )  # When placed in an environment, these are added/removed

    def set_environment(self, environment):
        for obstacle in self.current_obstacles:
            self.model.remove_constraint(obstacle)

        self.current_obstacles = environment.obstacles
        for obstacle in self.current_obstacles:
            self.model.add_constraint(obstacle)

    def sense(self, environment) -> dict[str, jnp.ndarray]:
        """Take sensor measurements for all sensors and return a dictionary of data.

        Returns:
            A dictionary of sensor data where each key is the sensor type."""
        sensor_data = {}
        for sensor in self.sensors:
            sensor_data.update(sensor.sense(self.state, environment))
        return sensor_data

    def estimate_state(self) -> jnp.ndarray:
        """Estimate the current state of the model."""
        return self.state

    def calculate_control(self, t: float) -> jnp.ndarray:
        """Calculate the control input based on the current state and time.

        Args:
            t: The current time.

        Returns:
            The control input as a jnp.ndarray."""
        return self.policy(self.estimate_state(), t)

    def step(self, t: float, dt: float) -> None:
        """Perform a single step of the actor's operation. Update the state and control.

        Args:
            t: The current time.
            dt: The time step for the model evolution.

        Returns:
            The next state of the model after applying the control policy."""
        self.control = self.calculate_control(t)
        next_state = self.model.discrete_dynamics(self.state, self.control, t, dt)
        self.state = next_state

    def draw(self, idx):
        """Draw the actor's model."""
        self.model.draw(self.state, self.control, idx)
