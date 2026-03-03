import numpy as np
import matplotlib.pyplot as plt

def unit(v):
    """
    単位ベクトル
    """
    nv = np.linalg.norm(v)
    return v / nv

def cross(u, v):
    """
    外積
    """
    return u[1] * v[0] - u[0] * v[1]

def normal(v):
    """
    法線ベクトル
    """
    return np.array([v[1], -v[0]], dtype=float)


def fillet(pt1, pt2, pt3, r, n=20):
    # prepare
    pt1 = np.asarray(pt1, dtype=float)
    pt2 = np.asarray(pt2, dtype=float)
    pt3 = np.asarray(pt3, dtype=float)
    r = float(r)

    u = pt1 - pt2
    v = pt3 - pt2

    e_u = unit(u)
    e_v = unit(v)

    cr = cross(e_u, e_v)
    turn = "left" if cr > 0 else "right"
    n_u = normal(e_u) if turn == "left" else -normal(e_u)
    n_v = -normal(e_v) if turn == "left" else normal(e_v)
    
    # calculate center
    p1 = pt2 + n_u*r
    p2 = pt2 + n_v*r

    s = cross(p2 - p1, e_v) / cross(e_u, e_v)
    center = p1 + e_u * s

    # tangent point
    t1 = center - n_u * r
    t2 = center - n_v * r

    # return arc points
    theta_1 = np.arctan2(t1[0] - center[0], t1[1] - center[1])
    theta_2 = np.arctan2(t2[0] - center[0], t2[1] - center[1])

    if turn == "right":
        if theta_2 <= theta_1:
            theta_2 += 2 * np.pi
    else:
        if theta_2 >= theta_1:
            theta_2 -= 2 * np.pi    
    ang = np.linspace(theta_1, theta_2, int(n))

    arc = np.column_stack([center[0] + r * np.sin(ang), center[1] + r * np.cos(ang)])

    dy = arc[1:, 0] - arc[:-1, 0]
    dx = arc[1:, 1] - arc[:-1, 1]
    psi = np.arctan2(dx, dy)
    psi = np.r_[psi, psi[-1]]

    return t1, t2, arc, psi, center

if __name__ == '__main__':
    WP =  np.array([
        [-3000.        , -1000.        ],
        [-2654.35793802, -1000.        ],
        [-1672.46500744, -1109.23541986],
        [ -683.45658059,  -152.5118353 ],
        [    0.        ,   -32.        ],
    ], dtype=float)
    arc_list = []

    for i in range(len(WP)-2):
        t1, t2, arc, psi, center = fillet(WP[i], WP[i+1], WP[i+2], r=500, n=20)
        plt.plot(arc[:, 1], arc[:, 0])
        plt.scatter(center[1], center[0], s=10, label=f"No. {i}")
        arc_list.append(arc)

    arcs = np.concatenate(arc_list, axis=0)
    pts = np.vstack([WP[0], arcs, WP[-1]])
    plt.plot(WP[:, 1], WP[:, 0], alpha=0.3)
    plt.plot(pts[:, 1], pts[:, 0])
    plt.axis("equal")
    plt.legend()
    plt.show()