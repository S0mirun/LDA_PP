import numpy as np
import matplotlib.pyplot as plt


class RandomWindSpeedMaki:
    """Random wind speed generation module besed on maki's practical method.

    References:
        - Maki, A., Maruyama, Y., Dostal, L. et al. Practical method for evaluating wind influence on autonomous ship operations. J Mar Sci Technol 27, 1302-1313 (2022). https://doi.org/10.1007/s00773-022-00901-w
    """

    def __init__(self, f_ref: float = 0.5) -> None:
        """Constructor

        Args:
            f_ref (float, optional): reference frequency. Defaults to 0.5.
        """
        self.f_ref = f_ref

    def reset(self, u: float):
        """Reset functions

        Args:
            u (float): Initial true wind speed

        Returns:
            float: True wind speed
        """
        # initialize
        u = check_plus(u)
        self.u, self.u_0, self.u_bar = u, u, u
        self.t = 0.0
        self.I_Nt_u = 0.0
        self.alpha, self.beta, self.sigma = self._filtter_coeff(self.u_bar)
        return u

    def _filtter_coeff(self, u_10: float):
        """Computes fillter coefficients

        Args:
            u_10 (float): wind speed

        Returns:
            list[float]: alpha, beta, sigma
        """
        ### Davenport and Hino ###
        Z = 15.0
        Kfriction = 0.001
        alpha_hino = 1.0 / 8.0
        m = 2
        u_bar = np.sqrt(6.0 * Kfriction * u_10**2)
        term1 = u_10 * alpha_hino / np.sqrt(Kfriction)
        term2 = (Z / 10.0) ** (2 * m * alpha_hino - 1.0)
        beta_hino = 1.169 * 1.0e-3 * term1 * term2
        ### Linear filter with Hino's spectrum ###
        #   Asymptotic value at f = 0
        Suw_H_0 = 0.2382 * u_bar**2 / beta_hino
        #   Asymptotic value at f = f_ref
        Suw_H_f_ref = Suw_H_0 * (1.0 + (self.f_ref / beta_hino) ** 2) ** (-5.0 / 6.0)
        #
        f_ref2 = self.f_ref**2
        pi2 = 2.0 * np.pi
        alpha2 = f_ref2 * pi2**2 * Suw_H_f_ref / (Suw_H_0 - Suw_H_f_ref)
        beta2 = Suw_H_0 * alpha2 / pi2
        #
        alpha = np.sqrt(alpha2)
        beta = np.sqrt(beta2)
        sigma = np.sqrt(2 * np.pi) * beta
        return alpha, beta, sigma

    def get_time(self):
        return self.t

    def get_state(self):
        return self.u

    def step(self, dt: float, np_random: np.random.Generator = None):
        """One step calculation of SDE

        Args:
            dt (float): time step
            np_random (np.random.Generator, optional): Generator of random number. Defaults to None.

        Returns:
            float: True wind speed in next step
        """
        if np_random is None:
            np_random = np.random
        # get random number
        dW = np.sqrt(dt) * np_random.normal()
        #
        t = self.t
        I_Nt_u = self.I_Nt_u
        # solve SDE
        t_n = t + dt
        I_Nt_u_n = ito_integral(I_Nt_u, dt, dW, self.alpha, self.beta)
        u_n = ornstein_uhlenbeck_process(
            t_n, self.u_0, self.u_bar, I_Nt_u_n, self.alpha
        )
        # update
        self.t = t_n
        self.u = u_n
        self.I_Nt_u = I_Nt_u_n
        return self.u


class RandomWindSpeedEM:
    """Random wind speed generation module besed on Euler Maruyama's method."""

    def __init__(self, f_ref: float = 0.5) -> None:
        """Constructor

        Args:
            f_ref (float, optional): reference frequency. Defaults to 0.5.
        """
        self.f_ref = f_ref

    def reset(self, u):
        """Reset functions

        Args:
            u (float): Initial true wind speed

        Returns:
            float: True wind speed
        """
        # initialize
        u = check_plus(u)
        self.u, self.u_0, self.u_bar = u, u, u
        self.t = 0.0
        self.alpha, self.beta, self.sigma = self._filtter_coeff(self.u_bar)
        return u

    def _filtter_coeff(self, u_10: float):
        """Computes fillter coefficients

        Args:
            u_10 (float): wind speed

        Returns:
            list[float]: alpha, beta, sigma
        """
        ### Davenport and Hino ###
        Z = 15.0
        Kfriction = 0.001
        alpha_hino = 1.0 / 8.0
        m = 2
        u_bar = np.sqrt(6.0 * Kfriction * u_10**2)
        term1 = u_10 * alpha_hino / np.sqrt(Kfriction)
        term2 = (Z / 10.0) ** (2 * m * alpha_hino - 1.0)
        beta_hino = 1.169 * 1.0e-3 * term1 * term2
        ### Linear filter with Hino's spectrum ###
        #   Asymptotic value at f = 0
        Suw_H_0 = 0.2382 * u_bar**2 / beta_hino
        #   Asymptotic value at f = f_ref
        Suw_H_f_ref = Suw_H_0 * (1.0 + (self.f_ref / beta_hino) ** 2) ** (-5.0 / 6.0)
        #
        f_ref2 = self.f_ref**2
        pi2 = 2.0 * np.pi
        alpha2 = f_ref2 * pi2**2 * Suw_H_f_ref / (Suw_H_0 - Suw_H_f_ref)
        beta2 = Suw_H_0 * alpha2 / pi2
        #
        alpha = np.sqrt(alpha2)
        beta = np.sqrt(beta2)
        sigma = np.sqrt(2 * np.pi) * beta
        return alpha, beta, sigma

    def get_time(self):
        return self.t

    def get_state(self):
        return self.u

    def step(self, dt, np_random=None):
        """One step calculation of SDE

        Args:
            dt (float): time step
            np_random (np.random.Generator, optional): Generator of random number. Defaults to None.

        Returns:
            float: True wind speed in next step
        """
        if np_random is None:
            np_random = np.random
        # get random number
        dW = np.sqrt(dt) * np_random.normal()
        #
        t = self.t
        # solve SDE
        t_n = t + dt
        du = -self.alpha * (self.u - self.u_bar) * dt + self.sigma * dW
        u_n = self.u + du
        self.u = u_n
        # update
        self.t = t_n
        self.u = u_n
        return self.u


class RandomWindDirectionEM:
    """Random wind direction generation module

    References:
        - ???
    """

    def __init__(self, sigma_dir: float = 2.3) -> None:
        """Constructor

        Args:
            sigma_dir (float, optional): ???. Defaults to 2.3.
        """
        self.sigma_dir = sigma_dir
        pass

    def reset(self, gamma: float, u_bar: float):
        """Reset functions

        Args:
            gamma (float): Initial true wind direction
            u_bar (float): Mean true wind speed

        Returns:
            float: True wind direction
        """
        # initialize
        gamma = gamma % (2 * np.pi)
        self.gamma_bar = gamma
        self.gamma_ = 0
        self.gamma = self.gamma_ + self.gamma_bar
        self.u_bar = u_bar
        self.t = 0.0
        self.alpha, self.beta, self.sigma = self._filtter_coeff(self.u_bar)
        return gamma

    def _filtter_coeff(self, u_10: float):
        """Computes fillter coefficients

        Args:
            u_10 (float): wind speed

        Returns:
            list[float]: alpha, beta, sigma
        """
        alpha = (self.sigma_dir**2) * (u_10 ** (3.0 / 2.0)) / (2.0 * 32.0**2)
        beta2 = self.sigma_dir**2 / (2 * np.pi)
        beta = np.sqrt(beta2) * np.pi / 180
        sigma = self.sigma_dir * np.pi / 180
        return alpha, beta, sigma

    def get_time(self):
        return self.t

    def get_state(self):
        return self.gamma

    def step(self, dt: float, np_random: np.random.Generator = None):
        """One step calculation of SDE

        Args:
            dt (float): time step
            np_random (np.random.Generator, optional): Generator of random number. Defaults to None.

        Returns:
            float: True wind speed in next step
        """
        if np_random is None:
            np_random = np.random
        # get random number
        dW = np.sqrt(dt) * np_random.normal()
        #
        t = self.t
        # solve SDE
        t_n = t + dt
        dgamma_ODE = -self.alpha * np.sin(self.gamma_ - 0.0)
        dgamma_SDE = self.sigma
        dgamma = dgamma_ODE * dt + dgamma_SDE * dW
        gamma_n = self.gamma_ + dgamma
        self.gamma_ = gamma_n
        # update
        self.t = t_n
        self.gamma = gamma_n + self.gamma_bar
        return self.gamma


def check_plus(var: float):
    if var <= 0.0:
        return 1.0e-16
    return var


def ornstein_uhlenbeck_process(
    t: float,
    x_0: float,
    x_bar: float,
    It: float,
    alpha: float,
):
    exp = np.exp(-alpha * t)
    x = x_0 * exp + x_bar * (1.0 - exp) + It
    return x


def ito_integral(
    I_Nt: float,
    dt: float,
    dW: float,
    alpha: float,
    beta: float,
):
    sigma = np.sqrt(2.0 * np.pi) * beta
    I_Nt_n = np.exp(-alpha * dt) * (I_Nt + sigma * dW)
    return I_Nt_n


if __name__ == "__main__":
    # parameter
    dt = 0.01
    t = 0
    T = 1000
    u_0 = 10.0  # (m/s)
    gamma_0 = 0  # (rad.)
    # simuation
    rwgen_u_maki = RandomWindSpeedMaki()
    u_maki = [rwgen_u_maki.reset(u_0)]
    t = [rwgen_u_maki.get_time()]
    while rwgen_u_maki.get_time() <= T:
        u_maki.append(rwgen_u_maki.step(dt))
        t.append(rwgen_u_maki.get_time())

    rwgen_u_em = RandomWindSpeedEM()
    u_em = [rwgen_u_em.reset(u_0)]
    while rwgen_u_em.get_time() <= T:
        u_em.append(rwgen_u_em.step(dt))

    rwgen_gamma_em = RandomWindDirectionEM()
    gamma_em = [rwgen_gamma_em.reset(gamma_0, u_0) * 180 / np.pi]
    while rwgen_gamma_em.get_time() <= T:
        gamma_em.append(rwgen_gamma_em.step(dt) * 180 / np.pi)
    # plot
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(t, u_maki, label="Maki")
    axes[0].plot(t, u_em, label="Euler Maruyama")
    axes[0].legend()
    axes[0].set_ylabel("$u \ \mathrm{(m/s)}$")
    axes[1].plot(t, gamma_em, label="Euler Maruyama")
    axes[1].legend()
    axes[1].set_ylabel("$\\gamma \ \mathrm{(deg.)}$")
    axes[1].set_xlabel("$t \ \mathrm{(s)}$")
    plt.show()
