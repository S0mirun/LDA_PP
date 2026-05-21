import os

import numpy as np
import matplotlib.pyplot as plt

from utils.PP.dictionary_of_port import dictionary

DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]


def source_flow(z, z0, Q):
    """ 吹き出し流れ """
    r = z - z0
    if abs(r) < 1e-8:
        return 0
    return Q / (2*np.pi*abs(r)) * (r / abs(r))


def vortex_flow(z, z0, Gamma):
    """ 渦流れ """
    r = z - z0
    if abs(r) < 1e-8:
        return 0
    return Gamma / (2*np.pi*abs(r)) * ((-1j) * r / abs(r))


def uniform_flow(z, U, angle=0):
    """ 一様流 """
    return U * np.exp(1j * angle)


def set_flows(flows, source_params=None, vortex_params=None):
    flows += [
        lambda z, z0=p["z0"], Q=p["Q"]: source_flow(z, z0, Q)
        for p in source_params
    ]

    flows += [
        lambda z, z0=p["z0"], Gamma=p["Gamma"]: vortex_flow(z, z0, Gamma)
        for p in vortex_params
    ]

    return flows


if __name__ == '__main__':
    SAVE_DIR = f"{DIR}/../../outputs/{dirname}"
    # os.makedirs(SAVE_DIR, exist_ok=True)

    port_number: int = 10
    port = dictionary()[port_number]
    # 0: Osaka_1A, 1: Tokyo_2C, 2: Yokkaichi_2B, 3: Sakaide, 4: Osaka_1B
    # 5: Else_2, 6: Kashima, 7: Aomori, 8: Hachinohe, 9: Shimizu
    # 10: Tomakomai, 11: KIX

    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.linspace(-10, 10, 40)
    y = np.linspace(-10, 10, 40)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    flows = []
    source_params = [
        {"z0": 0 - 0j,    "Q": -50.0}, # berth
    ]
    # source_params = [
    #     {"z0": -9.5 - 9.5j,    "Q": -50.0}, # berth
    # ]

    vortex_params = []
    # vortex_params = [
    #     {"z0": 0 + 2j,    "Gamma": 10.0},
    #     {"z0": 0 - 2j,    "Gamma": -10.0},
    #     {"z0": 2 + 2j,    "Gamma": 10.0},
    #     {"z0": 2 - 2j,    "Gamma": -10.0},
    #     {"z0": -2 + 2j,    "Gamma": 10.0},
    #     {"z0": -2 - 2j,    "Gamma": -10.0},
    # ]
    # vortex_params = [
        # {"z0": -5 + 2.5j,    "Gamma": 20.0},
        # {"z0":  0 + 5j,      "Gamma": 20.0},
        # {"z0":  9 + 11j,     "Gamma": 20.0},
        # {"z0": -7.5 - 4j,    "Gamma": -20.0},
        # {"z0": -5 - 2.5j,    "Gamma": -20.0},
        # {"z0":  3 + 3j,      "Gamma": -20.0},
        # {"z0": 11 + 8j,      "Gamma": -20.0},
    # ]

    flows = set_flows(flows, source_params, vortex_params)

    for i in range(X.shape[0]):
        for k in range(X.shape[1]):
            z = X[i, k] + 1j * Y[i, k]
            velocity = sum(flow(z) for flow in flows)

            U[i, k] = velocity.real
            V[i, k] = velocity.imag

    ax.quiver(X, Y, U, V,
              angles='xy', scale_units='xy', scale=4.5)
    
    for p in vortex_params:
        z0 = p["z0"]
        Gamma = p["Gamma"]

        color = "green" if Gamma < 0 else "red"
        ax.scatter(z0.real, z0.imag, 
                color=color, s=10, marker="o", zorder=5)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()