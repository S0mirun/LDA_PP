import numpy as np
import numpy.typing as npt
import pandas as pd

from .ship import SurfaceShip
from .world import World
from .world import polarTrue2polarApparent as pT2pA
from .logger import Logger
from .viewer import Viewer
from .utils import Variables


class ManeuveringSimulation:
    def __init__(
        self,
        ship: SurfaceShip,
        world: World,
        dt_act: float = 1.0,
        dt_sim: float = 0.1,
        check_collide: bool = False,
    ):
        """Maneuvering Simulator module

        Args:
            ship (SurfaceShip): instance of ship module.
            world (World): instance of world module.
            dt_act (float, optional): Timestep of action. Defaults to 1.0.
            dt_sim (float, optional): Timestep of numerical integration. Defaults to 0.1.
            check_collide (bool, optional): Handle whether to use the collision detection module. Defaults to False.
        """
        self.ship = ship
        self.world = world
        self.dt_act = dt_act
        self.dt_sim = dt_sim
        self.check_collide = check_collide
        # log
        self.logger = Logger()
        self.viewer = Viewer(ship, world, self.logger)
        # time
        self.TIME_NAME = ["t [s]"]
        # state
        self.STATE_NAME = self.ship.STATE_NAME + self.world.STATE_NAME
        self.STATE_UPPER_BOUND = self.ship.STATE_UPPER_BOUND
        self.STATE_UPPER_BOUND += self.world.STATE_UPPER_BOUND
        self.STATE_LOWER_BOUND = self.ship.STATE_LOWER_BOUND
        self.STATE_LOWER_BOUND += self.world.STATE_LOWER_BOUND
        self.STATE_DIM = len(self.STATE_NAME)
        self.ship_state_idx = self.get_state_id(self.ship.STATE_NAME)
        self.world_state_idx = self.get_state_id(self.world.STATE_NAME)
        # observation
        self.OBSERVATION_NAME = self.ship.OBSERVATION_NAME + self.world.OBSERVATION_NAME
        self.OBSERVATION_UPPER_BOUND = self.ship.OBSERVATION_UPPER_BOUND
        self.OBSERVATION_UPPER_BOUND += self.world.OBSERVATION_UPPER_BOUND
        self.OBSERVATION_LOWER_BOUND = self.ship.OBSERVATION_LOWER_BOUND
        self.OBSERVATION_LOWER_BOUND += self.world.OBSERVATION_LOWER_BOUND
        self.OBSERVATION_DIM = len(self.OBSERVATION_NAME)
        # action
        self.ACTION_NAME = []
        self.ACTION_UPPER_BOUND = []
        self.ACTION_LOWER_BOUND = []
        for i, name in enumerate(self.ship.ACTION_NAME):
            if name not in self.world.STATE_NAME:
                self.ACTION_NAME.append(name)
                self.ACTION_UPPER_BOUND.append(self.ship.ACTION_UPPER_BOUND[i])
                self.ACTION_LOWER_BOUND.append(self.ship.ACTION_LOWER_BOUND[i])
        self.ACTION_DIM = len(self.ACTION_NAME)

    def reset(self, state: npt.ArrayLike, seed: int | np.random.Generator = None):
        """reset simulator

        Args:
            state (npt.ArrayLike): Initial condition of state variables
            seed (int or np.random.Generator, optional): random seed. Defaults to None.

        Returns:
            ObsType: observation variables
        """
        # np_random
        if seed is None:
            if not hasattr(self, "np_random"):
                seed = 100
                seed_seq = np.random.SeedSequence(seed)
                self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        elif type(seed) == int:
            seed_seq = np.random.SeedSequence(seed)
            self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        else:
            self.np_random = seed
        # split
        state = np.array(state)
        assert len(state) == self.STATE_DIM
        ship_state = state[self.ship_state_idx]
        world_state = state[self.world_state_idx]
        # initialization
        self.t = 0.0
        self.ship.reset(ship_state)
        self.world.reset(world_state)
        self.state = np.concatenate([ship_state, world_state])
        ship_observation = self.ship.observe_state(np_random=self.np_random)
        world_observation = self.world.observe_state(np_random=self.np_random)
        self.observation = np.concatenate([ship_observation, world_observation])
        # log
        header = []
        header += self.TIME_NAME
        header += self.STATE_NAME
        header += self.ACTION_NAME
        header += self.OBSERVATION_NAME
        header += ["collision"]
        self.logger.reset(header)
        return self.observation

    def step(
        self, action: npt.ArrayLike, eval_next_step: bool = False
    ) -> tuple[Variables, bool, dict]:
        """one step of simulation

        Args:
            action (Variables): Action variables
            eval_next_step (bool, optional): If you want to evaluate next step state, change this to True. Defaults to False.

        Returns:
            tuple[Variables, bool, dict]: Observation variable for the next step, handle to determine the end of simulation, and additional infomation
        """
        action = np.array(action)
        t, state, observation = self.t, self.state, self.observation
        # start loop
        terminated = False
        steps = int(self.dt_act / self.dt_sim + 0.5)
        for _ in range(steps):
            ##### current ##################################################
            # split state
            ship_state = state[self.ship_state_idx]
            world_state = state[self.world_state_idx]
            ship_action = np.concatenate([action, world_state])
            ### check collision ###
            if self.check_collide and (not terminated):
                ship_poly = self.ship.ship_shape(ship_state)
                terminated = terminated or self.world.check_collision(ship_poly)
            ### logging ###
            log = [t]
            log += state.tolist()
            log += action.tolist()
            log += observation.tolist()
            log += [int(terminated)]
            self.logger.append(log)
            ##### next #####################################################
            ### time ###
            t_n = t + self.dt_sim
            ### state ###
            world_state_n = self.world.step(self.dt_sim, np_random=self.np_random)
            ship_state_n = self.ship.step(ship_action, self.dt_sim)
            state_n = np.concatenate([ship_state_n, world_state_n])
            ### observation ###
            ship_observation_n = self.ship.observe_state(np_random=self.np_random)
            world_observation_n = self.world.observe_state(np_random=self.np_random)
            observation_n = np.concatenate([ship_observation_n, world_observation_n])
            ### update ###
            t = t_n
            state, observation = state_n, observation_n
        # logging last state
        if eval_next_step:
            ### check collision ###
            ship_state = state[self.ship_state_idx]
            if self.check_collide and (not terminated):
                ship_poly = self.ship.ship_shape(ship_state)
                terminated = terminated or self.world.check_collision(ship_poly)
            ### logging ###
            log = [t]
            log += state.tolist()
            log += action.tolist()
            log += observation.tolist()
            log += [terminated]
            self.logger.append(log)
        # postprocess for next step
        self.t, self.state, self.observation = t, state, observation
        info = {"t": self.t, "state": self.state, "observation": self.observation}
        return observation, terminated, info

    def step_with_world_state(
        self,
        action: Variables,
        world_state: Variables,
        eval_next_step: bool = False,
    ) -> tuple[Variables, bool, dict]:
        """one step of simulation

        Args:
            action (Variables): Action variables
            world_state (Variables): World state variables. (eg. wind)
            eval_next_step (bool, optional): If you want to evaluate next step state, change this to True. Defaults to False.

        Returns:
            tuple[Variables, bool, dict]: Observation variable for the next step, handle to determine the end of simulation, and additional infomation
        """
        t, state, observation = self.t, self.state, self.observation
        # start loop
        terminated = False
        steps = int(self.dt_act / self.dt_sim + 0.5)
        for _ in range(steps):
            ##### current ##################################################
            # split state
            ship_state = state[self.ship_state_idx]
            ship_action = np.concatenate([action, world_state])
            ### check collision ###
            if self.check_collide and (not terminated):
                ship_poly = self.ship.ship_shape(ship_state)
                terminated = terminated or self.world.check_collision(ship_poly)
            ### logging ###
            log = [t]
            log += state.tolist()
            log += action.tolist()
            log += observation.tolist()
            log += [terminated]
            self.logger.append(log)
            ##### next #####################################################
            ### time ###
            t_n = t + self.dt_sim
            ### state ###
            # world
            world_state_n = world_state
            # ship
            ship_state_n = self.ship.step(ship_action, self.dt_sim)
            state_n = np.concatenate([ship_state_n, world_state_n])
            ### observation ###
            ship_observation_n = self.ship.observe_state(np_random=self.np_random)
            world_observation_n = self.world.observe_state(np_random=self.np_random)
            observation_n = np.concatenate([ship_observation_n, world_observation_n])
            ### update ###
            t = t_n
            state, observation = state_n, observation_n
        # logging last state
        if eval_next_step:
            state[self.world_state_idx] = np.zeros_like(world_state)
            ### check collision ###
            ship_state = state[self.ship_state_idx]
            if self.check_collide and (not terminated):
                ship_poly = self.ship.ship_shape(ship_state)
                terminated = terminated or self.world.check_collision(ship_poly)
            ### logging ###
            log = [t]
            log += state.tolist()
            log += action.tolist()
            log += observation.tolist()
            log += [int(terminated)]
            self.logger.append(log)
        # postprocess for next step
        self.t, self.state, self.observation = t, state, observation
        info = {"t": self.t, "state": self.state, "observation": self.observation}
        return observation, terminated, info

    def log2df(self) -> pd.DataFrame:
        df = self.logger.get_df()
        # add apparent wind
        U_T = df["true_wind_speed [m/s]"].to_numpy()
        gamma_T = df["true_wind_direction [rad]"].to_numpy()
        U_T_hat = df["true_wind_speed_hat [m/s]"].to_numpy()
        gamma_T_hat = df["true_wind_direction_hat [rad]"].to_numpy()
        u = df["u_velo [m/s]"].to_numpy()
        vm = df["vm_velo [m/s]"].to_numpy()
        psi = df["psi [rad]"].to_numpy()
        U_A, gamma_A = pT2pA(U_T, gamma_T, u, vm, psi)
        df["apparent_wind_speed [m/s]"] = U_A
        df["apparent_wind_direction [rad]"] = gamma_A
        df["apparent_wind_speed_hat [m/s]"] = U_A + (U_T_hat - U_T)
        df["apparent_wind_direction_hat [rad]"] = gamma_A + (gamma_T_hat - gamma_T)
        return df

    def log2csv(self, dir: str = "./", fname: str = "test"):
        self.logger.to_csv(
            dir=dir,
            fname=fname,
            df=self.log2df(),
        )

    def log2img(self, dir: str = "./", fname: str = "test", ext_type: str = "png"):
        self.viewer.get_x0y0_plot(
            dir=dir,
            fname=f"{fname}_trajectory",
            ext_type=ext_type,
        )
        self.viewer.get_timeseries_plot(
            self.STATE_NAME,
            dir=dir,
            fname=f"{fname}_state_timeseries",
            ext_type=ext_type,
        )
        self.viewer.get_timeseries_plot(
            self.OBSERVATION_NAME,
            dir=dir,
            fname=f"{fname}_observation_timeseries",
            ext_type=ext_type,
        )
        self.viewer.get_timeseries_plot(
            self.ACTION_NAME,
            dir=dir,
            fname=f"{fname}_action_timeseries",
            ext_type=ext_type,
        )

    def get_time(self) -> float:
        return self.t

    def get_state(self) -> Variables:
        return self.state

    def observe_state(self) -> Variables:
        return self.observation

    @staticmethod
    def get_variables_id(variables: list[str], names: str):
        return [variables.index(name) for name in names]

    def get_state_id(self, names: str):
        return self.get_variables_id(self.STATE_NAME, names)

    def get_observation_id(self, names: str):
        return self.get_variables_id(self.OBSERVATION_NAME, names)

    def get_action_id(self, names: str):
        return self.get_variables_id(self.ACTION_NAME, names)
