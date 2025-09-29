import matplotlib.pyplot as plt

from .ship import SurfaceShip
from .world import World
from .logger import Logger


class Viewer:
    def __init__(
        self,
        ship: SurfaceShip,
        world: World,
        logger: Logger,
    ):
        self.ship = ship
        self.world = world
        self.logger = logger

    def get_x0y0_plot(
        self, dir: str = "./", fname: str = "test", ext_type: str = "png"
    ):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
        ax.set_xlabel("$y_{0}/L_{\mathrm{pp}}$")
        ax.set_ylabel("$x_{0}/L_{\mathrm{pp}}$")
        ax.axis("equal")
        # load results
        Lpp = self.ship.L
        df = self.logger.get_df()
        # trajectory
        x_0 = df[[self.ship.STATE_NAME[0]]].to_numpy()
        y_0 = df[[self.ship.STATE_NAME[2]]].to_numpy()
        kwargs = {"color": "black", "ls": "dashed", "lw": 0.5}
        ax.plot(y_0 / Lpp, x_0 / Lpp, **kwargs)
        # ship
        ship_states = df[self.ship.STATE_NAME].to_numpy()
        ship_polys = []
        for i in range(len(df) - 1, -1, -100):
            ship_poly = self.ship.ship_appearance(ship_states[i])
            ship_poly = [ship_poly] if ship_poly is not list else ship_poly
            for ship_poly_ in ship_poly:
                ship_polys.append(ship_poly_[:, [1, 0]] / Lpp)
        kwargs = {"fill": False, "lw": 0.3, "ec": "black"}
        for ship_poly in ship_polys:
            polygon = plt.Polygon(ship_poly, **kwargs)
            ax.add_patch(polygon)
        # world
        kwargs = {"fill": True, "lw": 0.3, "fc": "#C8CCDE", "ec": "#0A0F60"}
        for world_poly in self.world.OBSTACLE_POLYGONS:
            world_poly_ = world_poly[:, [1, 0]] / Lpp
            polygon = plt.Polygon(world_poly_, **kwargs)
            ax.add_patch(polygon)
        #
        y_min = 1.2 * min(y_0.min() - y_0.mean(), -Lpp / 2) + y_0.mean()
        y_max = 1.2 * max(y_0.max() - y_0.mean(), Lpp / 2) + y_0.mean()
        ax.set_xlim(y_min / Lpp, y_max / Lpp)
        x_min = 1.2 * min(x_0.min() - x_0.mean(), -Lpp / 2) + x_0.mean()
        x_max = 1.2 * max(x_0.max() - x_0.mean(), Lpp / 2) + x_0.mean()
        ax.set_ylim(x_min / Lpp, x_max / Lpp)
        #
        fig.savefig(f"{dir}/{fname}.{ext_type}")
        plt.clf()
        plt.close()

    def get_timeseries_plot(
        self,
        NAMES: str | list[str],
        dir: str = "./",
        fname: str = "test",
        ext_type="png",
    ):

        NAMES = [NAMES] if NAMES is str else NAMES
        # load results
        df = self.logger.get_df()
        T = df.index[-1] - df.index[0]
        D = len(NAMES)
        #
        kwargs = {"sharex": True, "figsize": (0.1 * T, 2 * D), "tight_layout": True}
        fig, axes = plt.subplots(D, 1, **kwargs)
        axes = [axes] if D == 1 else axes
        for i in range(D):
            axes[i].set_ylabel(NAMES[i])
        axes[-1].set_xlabel("t [s]")
        # plot
        for i in range(D):
            kwargs = {"color": "black", "linestyle": "solid"}
            axes[i].plot(df.index, df[NAMES[i]], **kwargs)
        #
        fig.savefig(f"{dir}/{fname}.{ext_type}")
        plt.clf()
        plt.close()

    # @staticmethod
    # def get_colors(N=1):
    #     BASE_COLORS = ["black", "red", "blue"]
    #     colors = []
    #     for i in range(N):
    #         ii = i % len(BASE_COLORS)
    #         colors.append(BASE_COLORS[ii])
    #     return colors

    # @staticmethod
    # def get_linestyles(N=1):
    #     BASE_LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]
    #     linestyles = []
    #     for i in range(N):
    #         ii = i % len(BASE_LINESTYLES)
    #         linestyles.append(BASE_LINESTYLES[ii])
    #     return linestyles
