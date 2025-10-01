import numpy as np


__all__ = [
    "first_order_delay_ode_rhs",
    "linear_delay_ode_rhs",
]


def first_order_delay_ode_rhs(y, u, T, K):
    """Right hand side of ordinary differential equations for first-order delay systems
    ODE : `dy/dt = K/T*u(t) - 1/T*y(t)`

    Args:
        y (float or ndarray): Output variables
        u (float or ndarray): Input variables
        T (float): time constant
        K (float): gain
    """
    dydt = (K * u - y) / T
    return dydt


def linear_delay_ode_rhs(y, u, K):
    """Right hand side of ordinary differential equations for linear delay systems

    Args:
        y (float or ndarray): Output variables
        u (float or ndarray): Input variables
        K (float): Constant rate of output variables
    """
    epsilon = 0.1
    dydt = np.clip((u - y) / epsilon, -K, K)
    # dydt = K * np.tanh((u - y) / (K * epsilon))
    # dydt = K * np.sign(u - y)
    return dydt
