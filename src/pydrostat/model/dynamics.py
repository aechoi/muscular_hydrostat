from abc import ABC, abstractmethod

from jax import vmap, jacobian, lax
import jax.numpy as jnp


class DynamicModel(ABC):
    """Dynamic model of a state space system
    TODO add back in autograded linearization"""

    def __init__(self):
        # self.dt_jac_state = jacobian(self.discrete_dynamics, 0)
        # self.dt_jac_control = jacobian(self.discrete_dynamics, 1)
        pass

    @abstractmethod
    def continuous_dynamics(self, state, control, t):
        raise NotImplementedError

    def discrete_dynamics(self, state, control, t, dt):
        return self.integrator_rk(state, control, t, dt, self.continuous_dynamics)

    def integrator_rk(self, state, control, t, dt):
        k1 = self.continuous_dynamics(state, control, t)
        k2 = self.continuous_dynamics(state + k1 / 2 * dt, control, t)
        k3 = self.continuous_dynamics(state + k2 / 2 * dt, control, t)
        k4 = self.continuous_dynamics(state + k3 * dt, control, t)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(
        self, init_state: jnp.ndarray, policy: callable, final_step: int, dt: float
    ):
        """Simulate the system under some policy

        Args:
            init_state: the initial state
            policy: a function that takes in a state and time
            final_step: the number of total simulated time steps (including initial)
            dt: the time interval between samples"""

        def step_fn(carry, t):
            state = carry
            control = policy(state, t)
            next_state = self.discrete_dynamics(state, control, t, dt)
            return next_state, (next_state, control)

        times = jnp.arange(final_step) * dt
        _, (state_traj, control_traj) = lax.scan(step_fn, init_state, times)

        return state_traj, control_traj

    # def _linearize(self, state, control, t, dt):
    #     """Return the linear dynamic approximation about a particular state, control,
    #     and time."""
    #     A = self.dt_jac_state(state, control, t, dt)
    #     B = self.dt_jac_control(state, control, t, dt)
    #     C = self.discrete_dynamics(state, control, t, dt) - A @ state - B @ control

    #     return A, B, C

    # def linearize_traj(self, states, controls, ts, dt):
    #     """Return a sequence of linear dynamic approximations about a trajectory of
    #     states and controls at multiple times."""
    #     return vmap(self._linearize, in_axes=[0, 0, 0, None])(
    #         states[:-1], controls, ts, dt
    #     )
