from . import f2py_mmg_esso_osaka_150m


class MMGModel(object):
    def __init__(self):
        pass

    def ode_rhs(self, x, u, w):
        [delta_rudder, n_prop, n_bt, n_st] = u
        [Wind_velocity, Wind_Direction] = w
        dx = f2py_mmg_esso_osaka_150m.esso_osaka_realscale.mmg_lowspeed_model(
            x,
            delta_rudder,
            n_prop,
            n_bt,
            n_st,
            Wind_Direction,
            Wind_velocity,
        )
        return dx
