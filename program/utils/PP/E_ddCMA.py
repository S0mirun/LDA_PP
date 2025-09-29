# 新しい終了条件を追加したver. うまくいかなかったら削除する

import warnings
from collections import deque
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class DdCma:
    """dd-CMA: CMA-ES with diagonal decoding [1]
    Note
    ----
    If you are interested in constrained optimization and/or multi-fidelity optimization,
    check the following repository:
    https://github.com/akimotolab/multi-fidelity
    
    History
    -------
    2022/03/24: Mirroring box constraint handling and periodic variable handling [3] have been implemented.
    2020/06/10: Restart (IPOP mechanism) [2] has been implemented.
    2019/03/23: release
    Reference
    ---------
    [1] Y. Akimoto and N. Hansen. 
    Diagonal Acceleration for Covariance Matrix Adaptation Evolution Strategies
    Evolutionary Computation (2020) 28(3): 405--435.
    [2] A. Auger and N. Hansen.
    A Restart CMA Evolution Strategy With Increasing Population Size
    IEEE Congress on Evolutionary Computation (2005): 1769-1776.
    [3] Y. Yamaguchi and A. Akimoto.
    A Note on the CMA-ES for Functions with Periodic Variables
    Genetic and Evolutionary Computation Conference Companion (2018): 227-228.
    """
    
    def __init__(self, xmean0, sigma0, 
                 lam=None,
                 flg_covariance_update=True,
                 flg_variance_update=True,
                 flg_active_update=True,
                 beta_eig=None,
                 beta_thresh=2.,
                 seed=None):
        """
        Parameters
        ----------
        xmean0 : 1d array-like
            initial mean vector
            
        sigma0 : 1d array-like
            initial diagonal decoding
            
        lam : int, optional (default = None)
            population size
            (集団サイズ。デフォルトでは次元数Nに基づいて 4 + int(3 * math.log(N)) のように設定)
            ex) 探索範囲が8個あったら lam = 12
            
        flg_covariance_update : bool, optional (default = True)
            update C if this is True
            (共分散行列Cを更新するかどうかを指定するフラグ)
            
        flg_variance_update : bool, optional (default = True)
            update D if this is True
            (対角デコード行列Dを更新するかどうかを指定するフラグ)
            
        flg_active_update : bool, optional (default = True)
            update C and D with active update
            (アクティブな更新（良い解だけでなく悪い解も考慮する更新）を行うかどうかを指定するフラグ)
            
        beta_eig : float, optional (default = None)
            coefficient to control the frequency of matrix decomposition
            (行列分解(固有値分解)の頻度を制御する係数です。Noneの場合は次元数Nに基づいて10 * Nに設定)
            
        beta_thresh : float, optional (default = 2.)
            threshold parameter for beta control
            (betaの制御に関する閾値。betaは行列の更新率を調整する係数であり、beta_threshはbetaが調整される基準となります。閾値を大きくすると、より強い更新が必要な場合にのみbetaが変化)
        """
        self.N = len(xmean0)
        self.chiN = np.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N * self.N))

        # options
        self.flg_covariance_update = flg_covariance_update
        self.flg_variance_update = flg_variance_update
        self.flg_active_update = flg_active_update
        self.beta_eig = beta_eig if beta_eig else 10. * self.N
        self.beta_thresh = beta_thresh
        
        # parameters for recombination and step-size adaptation
        self.lam = lam if lam else 4 + int(3 * math.log(self.N)) # population size
        
        assert self.lam > 2
        w = math.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam+1))
        w[w > 0] /= np.sum(np.abs(w[w > 0]))
        w[w < 0] /= np.sum(np.abs(w[w < 0]))
        self.mueff_positive = 1. / np.sum(w[w > 0] ** 2)
        self.mueff_negative = 1. / np.sum(w[w < 0] ** 2)
        self.cm = 1.
        self.cs = (self.mueff_positive + 2.) / (self.N + self.mueff_positive + 5.)
        self.ds = 1. + self.cs + 2. * max(0., math.sqrt((self.mueff_positive - 1.) / (self.N + 1.)) - 1.)
        
        # parameters for covariance matrix adaptation
        expo = 0.75
        mu_prime = self.mueff_positive + 1. / self.mueff_positive - 2. + self.lam / (2. * self.lam + 10.)
        m = self.N * (self.N + 1) / 2
        self.cone = 1. / ( 2 * (m / self.N + 1.) * (self.N + 1.) ** expo + self.mueff_positive / 2.)
        self.cmu = min(1. - self.cone, mu_prime * self.cone)
        self.cc = math.sqrt(self.mueff_positive * self.cone) / 2.
        self.w = np.array(w)
        self.w[w < 0] *= min(1. + self.cone / self.cmu, 1. + 2. * self.mueff_negative / (self.mueff_positive + 2.))
        
        # parameters for diagonal decoding
        m = self.N
        self.cdone = 1. / ( 2 * (m / self.N + 1.) * (self.N + 1.) ** expo + self.mueff_positive / 2.)
        self.cdmu = min(1. - self.cdone, mu_prime * self.cdone)
        self.cdc = math.sqrt(self.mueff_positive * self.cdone) / 2.
        self.wd = np.array(w)
        self.wd[w < 0] *= min(1. + self.cdone / self.cdmu, 1. + 2. * self.mueff_negative / (self.mueff_positive + 2.))
        
        # dynamic parameters
        self.xmean = np.array(xmean0)
        self.D = np.array(sigma0)
        self.sigma = 1.
        self.C = np.eye(self.N)
        self.S = np.ones(self.N)
        self.B = np.eye(self.N)
        self.sqrtC = np.eye(self.N)
        self.invsqrtC = np.eye(self.N)
        self.Z = np.zeros((self.N, self.N))
        self.pc = np.zeros(self.N)
        self.pdc = np.zeros(self.N)
        self.ps = np.zeros(self.N)
        self.pc_factor = 0.
        self.pdc_factor = 0.
        self.ps_factor = 0.

        # others 
        self.teig = max(1, int(1. / (self.beta_eig * (self.cone + self.cmu))))
        self.neval = 0
        self.t = 0
        self.beta = 1.
        
        # strage for checker and logger
        self.arf = np.zeros(self.lam)
        self.arx = np.zeros((self.lam, self.N))
        self.idx = None
        
        # シード付き乱数生成器を追加(20250214)
        self.rng = np.random.default_rng(seed)

    def transform(self, z):
        y = np.dot(z, self.sqrtC) if self.flg_covariance_update else z
        return y * (self.D * self.sigma)

    def transform_inverse(self, y):
        z = y / (self.D * self.sigma)
        return np.dot(z, self.invsqrtC) if self.flg_covariance_update else z

    def sample(self):
        # シード値から、平均0、標準偏差1の正規分布に従うlam行N列のランダム行列を作成
        # arz = np.random.randn(self.lam, self.N)
        arz = self.rng.standard_normal((self.lam, self.N))
        # flg_covariance_updateがTrueの時にサンプルarzに共分散行列の特性を持たせる
        ary = np.dot(arz, self.sqrtC) if self.flg_covariance_update else arz
        # 実際のサンプル候補解arx
        arx = ary * (self.D * self.sigma) + self.xmean
        return arx, ary, arz

    def update(self, idx, arx, ary, arz):
        # shortcut
        w = self.w
        wc = self.w
        wd = self.wd
        sarz = arz[idx]
        sary = ary[idx]
        sarx = arx[idx]
        
        # recombination
        dz = np.dot(w[w > 0], sarz[w > 0])
        dy = np.dot(w[w > 0], sary[w > 0])
        self.xmean += self.cm * self.sigma * self.D * dy

        # step-size adaptation        
        self.ps_factor = (1 - self.cs) ** 2 * self.ps_factor + self.cs * (2 - self.cs)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff_positive) * dz
        normsquared = np.sum(self.ps * self.ps)
        hsig = normsquared / self.ps_factor / self.N < 2.0 + 4.0 / (self.N + 1)
        self.sigma *= math.exp((math.sqrt(normsquared) / self.chiN - math.sqrt(self.ps_factor)) * self.cs / self.ds)

        # C (intermediate) update
        # この部分では、探索方向の調整を行うため、共分散行列Cの更新項目をZに累積して記録しています。
        if self.flg_covariance_update:
            # Rank-mu
            # 候補解の分布に基づく共分散行列の更新。
            if self.cmu == 0:
                rank_mu = 0.
            elif self.flg_active_update:
                rank_mu = np.dot(sarz[wc>0].T * wc[wc>0], sarz[wc>0]) - np.sum(wc[wc>0]) * np.eye(self.N)
                rank_mu += np.dot(sarz[wc<0].T * (wc[wc<0] * self.N / np.linalg.norm(sarz[wc<0], axis=1) ** 2),
                                  sarz[wc<0]) - np.sum(wc[wc<0]) * np.eye(self.N)
            else:
                rank_mu = np.dot(sarz[wc>0].T * wc[wc>0], sarz[wc>0]) - np.sum(wc[wc>0]) * np.eye(self.N)
            # Rank-one
            # self.xmeanの更新方向（進んだ方向）に沿って調整する共分散行列の更新方式です。
            if self.cone == 0:
                rank_one = 0.
            else:
                self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff_positive) * self.D * dy 
                self.pc_factor = (1 - self.cc) ** 2 * self.pc_factor + hsig * self.cc * (2 - self.cc)
                zpc = np.dot(self.pc / self.D, self.invsqrtC)
                rank_one = np.outer(zpc, zpc) - self.pc_factor * np.eye(self.N)
            # Update
            # 更新項目（rank_muとrank_one）を組み合わせてself.Zに加え、Cの更新に役立てます
            # cmu(Rank-mu更新率)とcone(Rank-one更新率)は0から1の範囲で設定され、1に近いほど更新が強く、探索方向の適応が積極的に行われます。
            self.Z += (self.cmu * rank_mu + self.cone * rank_one)

        # D update
        # 変数ごとにスケールを調整する対角行列Dの更新を行います。
        if self.flg_variance_update:
            # Cumulation(self.pdcはself.Dの更新方向を追跡する"累積"ベクトル)
            self.pdc = (1 - self.cdc) * self.pdc + hsig * math.sqrt(self.cdc * (2 - self.cdc) * self.mueff_positive) * self.D * dy
            self.pdc_factor = (1 - self.cdc) ** 2 * self.pdc_factor + hsig * self.cdc * (2 - self.cdc)
            DD = self.cdone * (np.dot(self.pdc / self.D, self.invsqrtC) ** 2 - self.pdc_factor)
            # self.flg_active_updateがTrueの場合、評価値の良い解（正の重みwd > 0）と悪い解（負の重みwd < 0）の両方を考慮してDDを計算し、探索の広がりを調整します。
            if self.flg_active_update:
                # positive and negative update
                DD += self.cdmu * np.dot(wd[wd>0], sarz[wd>0] ** 2)
                DD += self.cdmu * np.dot(wd[wd<0] * self.N / np.linalg.norm(sarz[wd<0], axis=1)**2, sarz[wd<0]**2)
                DD -= self.cdmu * np.sum(wd)
            else:
                # positive update
                DD += self.cdmu * np.dot(wd[wd>0], sarz[wd>0] ** 2)
                DD -= self.cdmu * np.sum(wd[wd>0])
            if self.flg_covariance_update:
                self.beta = 1 / max(1, np.max(self.S) / np.min(self.S) - self.beta_thresh + 1.)
            else:
                self.beta = 1.
            self.D *= np.exp((self.beta / 2) * DD)

        # update C
        # この部分では、設定された周期teigごとに共分散行列Cを更新し、固有値分解を行って数値的に安定した状態を維持します。
        if self.flg_covariance_update and (self.t + 1) % self.teig == 0:
            D = np.linalg.eigvalsh(self.Z)
            fac = min(0.75 / abs(D.min()), 1.)
            self.C = np.dot(np.dot(self.sqrtC, np.eye(self.N) + fac * self.Z), self.sqrtC)            

            # force C to be correlation matrix
            cd = np.sqrt(np.diag(self.C))
            self.D *= cd
            self.C = (self.C / cd).T / cd

            # decomposition
            #np.linalg.eigh(self.C)は共分散行列Cの固有値分解を行う。この分解により、Cの固有値がDDに、固有ベクトルがself.Bに格納される
            DD, self.B = np.linalg.eigh(self.C)
            self.S = np.sqrt(DD)
            self.sqrtC = np.dot(self.B * self.S, self.B.T)
            self.invsqrtC = np.dot(self.B / self.S, self.B.T)
            # self.Zは0にリセットされ、次の更新周期に備えます。          
            self.Z[:, :] = 0.

    def onestep(self, func):
        """
        Parameter
        ---------
        func : callable
            parameter : 2d array-like with candidate solutions (x) as elements
            return    : 1d array-like with f(x) as elements
        """
        # sampling
        arx, ary, arz = self.sample()

        # evaluation(評価!!!)
        arf = func(arx)
        # arfの長さとは、集団の数(popサイズ)である。それぞれの集団に対して評価回数が呼び出されるためである。
        # よって10世代に1回ログを表示する設定にしているのでれば、評価関数の呼び出し回数は、popサイズ✖️10ずつ増えて表示される。
        self.neval += len(arf)
        
        # sort
        arf_idx = np.argsort(arf) # np.argsort(arf) は 評価値 (arf) を小さい順に並び替えたときのインデックスを取得
        
        # warnings.warn() を使って重複があることを警告する。エラーではないので、処理は止まらずに続行される
        if not np.all(arf[arf_idx[1:]] - arf[arf_idx[:-1]] > 0.):
            warnings.warn("assumed no tie, but there exists", RuntimeWarning)

        # update
        self.update(arf_idx, arx, ary, arz)

        # finalize
        self.t += 1
        self.arf = arf
        self.arx = arx
        self.idx = arf_idx
        
    # def upper_bounding_coordinate_std(self, coordinate_length):
    #     """Upper-bounding coordinate-wise standard deviation
        
    #     When some design variables are periodic, the coordinate-wise standard deviation 
    #     should be upper-bounded by r_i / 4, where r_i is the period of the ith variable.
    #     The correction of the overall covariance matrix, Sigma, is done as follows:
    #         Sigma = Correction * Sigma * Correction,
    #     where Correction is a diagonal matrix defined as
    #         Correction_i = min( r_i / (4 * Sigma_{i,i}^{1/2}), 1 ).
            
    #     In DD-CMA, the correction matrix is simply multiplied to D.
            
    #     For example, if a mirroring box constraint handling is used for a box constraint
    #     [l_i, u_i], the variables become periodic on [l_i - (u_i-l_i)/2, u_i + (u_i-l_i)/2]. 
    #     Therefore, the period is 
    #         r_i = 2 * (u_i - l_i).
        
    #     Parameters
    #     ----------
    #     coordinate_length : ndarray (1D) or float
    #         coordinate-wise search length r_i.
            
    #     References
    #     ----------
    #     T. Yamaguchi and Y. Akimoto. 
    #     A Note on the CMA-ES for Functions with Periodic Variables.
    #     GECCO '18 Companion, pages 227--228 (2018)
    #     """
    #     correction = np.fmin(coordinate_length / self.coordinate_std / 4.0, 1)
    #     self.D *= correction
        
        
    @property
    def coordinate_std(self):
        if self.flg_covariance_update:
            return self.sigma * self.D * np.sqrt(np.diag(self.C))
        else:
            return self.sigma * self.D

# DdCmaアルゴリズムの停止条件をチェックするためのクラスで、最適化が終了すべきタイミングを判断するさまざまな基準（BBOB基準）を提供
# bbobは"Black-Box Optimization Benchmarking"の略
class Checker:
    """BBOB termination Checker for dd-CMA"""
    def __init__(self, cma):
        assert isinstance(cma, DdCma)
        self._cma = cma
        self._init_std = self._cma.coordinate_std
        self._N = self._cma.N
        self._lam = self._cma.lam
        self._hist_fbest = deque(maxlen=10 + int(np.ceil(30 * self._N / self._lam)))
        self._hist_feq_flag = deque(maxlen=self._N)
        self._hist_fmin = deque()
        self._hist_fmed = deque()
        # ここで新たな属性を初期化しておく（初回チェック時に更新する方法でもよい）
        self._last_improvement_t = None  # 最後に改善があった世代
        self._last_improvement_value = None  # そのときの最小評価値

    def __call__(self):
        return self.bbob_check()

    #以下のメソッドは終了条件を満たしているかどうかのbool値を返す
    
    # 最大反復回数を超えたかを判定
    def check_maxiter(self):
        return self._cma.t > 100 + 50 * (self._N + 3) ** 2 / np.sqrt(self._lam)

    # 評価値（目的関数）の履歴の範囲が十分に小さいかどうかを確認
    # 条件を緩くしてもいいはず。ここを緩くしてリスタート増やすとか
    def check_tolhistfun(self):
        self._hist_fbest.append(np.min(self._cma.arf))
        return (self._cma.t >= 10 + int(np.ceil(30 * self._N / self._lam)) and
                np.max(self._hist_fbest) - np.min(self._hist_fbest) < 0.1) # さらに0.01から緩くした

    # 評価値が同一のサンプルが多い場合に、探索が収束しているとみなす
    def check_equalfunvals(self):
        k = int(math.ceil(0.1 + self._lam / 4))
        sarf = np.sort(self._cma.arf)
        self._hist_feq_flag.append(sarf[0] == sarf[k])
        return 3 * sum(self._hist_feq_flag) > self._N

    # 現在の標準偏差が初期値に対して小さすぎる場合に停止
    def check_tolx(self):
        return (np.all(self._cma.coordinate_std / self._init_std) < 0.1) #1e-12から変更 #さらに1e-3から緩くした(2/5)

    # 現在の標準偏差が初期値に比べて非常に大きくなりすぎた場合に停止
    def check_tolupsigma(self):
        return np.any(self._cma.coordinate_std / self._init_std > 1e3)

    # 評価値の変化が停滞している（進展がない）場合に停止
    def check_stagnation(self):
        self._hist_fmin.append(np.min(self._cma.arf))
        self._hist_fmed.append(np.median(self._cma.arf))
        _len = int(np.ceil(self._cma.t / 5 + 120 + 30 * self._N / self._lam))
        if len(self._hist_fmin) > _len:
            self._hist_fmin.popleft()
            self._hist_fmed.popleft()
        fmin_med = np.median(np.asarray(self._hist_fmin)[-20:])
        fmed_med = np.median(np.asarray(self._hist_fmed)[:20])
        return self._cma.t >= _len and fmin_med >= fmed_med

    # 共分散行列の条件数が大きすぎる場合に停止
    def check_conditioncov(self):
        return (np.max(self._cma.S) / np.min(self._cma.S) > 1e7
                or np.max(self._cma.D) / np.min(self._cma.D) > 1e7)

    # 軸方向で効果がなくなっている（探索が進展しなくなっている）場合に停止
    def check_noeffectaxis(self):
        t = self._cma.t % self._N
        test = 0.1 * self._cma.sigma * self._cma.D * self._cma.S[t] * self._cma.B[:, t]
        return np.all(self._cma.xmean == self._cma.xmean + test)

    # 座標方向で効果がなくなっている場合に停止
    def check_noeffectcoor(self):
        return np.all(self._cma.xmean == self._cma.xmean + 0.2 * self._cma.coordinate_std)

    # 評価値がすべて同じ場合に停止
    def check_flat(self):
        return np.max(self._cma.arf) == np.min(self._cma.arf)
    
    # 最小の評価値が次元×200世代更新されなければ、終了する
    def check_no_improvement_generations(self):
        """
        現在の最小評価値が、前回の改善以降 self._cma.N * 200 世代の間、
        1 以上の改善が見られなかった場合に True を返す。
        例えば、次元が 25 の場合、5000 世代連続で最小評価値が 1 以上改善されなければ停止。
        """
        current_best = np.min(self._cma.arf)
        # 初回呼び出し時は属性が None なので初期化する
        if self._last_improvement_t is None or self._last_improvement_value is None:
            self._last_improvement_t = self._cma.t
            self._last_improvement_value = current_best
            return False
        # 前回の改善値から 1 以上改善があった場合のみ、改善と見なし情報を更新する
        if (self._last_improvement_value - current_best) >= 1:
            self._last_improvement_value = current_best
            self._last_improvement_t = self._cma.t
            return False
        # 改善が見られなかった世代数が閾値以上の場合に停止
        if (self._cma.t - self._last_improvement_t) >= self._cma.N * 200:
            return True
        return False

    def bbob_check(self):
        if self.check_maxiter():
            return True, 'bbob_maxiter (最大反復回数を超えたため停止)'
        if self.check_tolhistfun():
            return True, 'bbob_tolhistfun (評価値の履歴の範囲が十分に小さいため停止)'
        if self.check_equalfunvals():
            return True, 'bbob_equalfunvals (同一評価値のサンプルが多いため停止)'
        if self.check_tolx():
            return True, 'bbob_tolx (標準偏差が小さすぎるため停止)'
        if self.check_tolupsigma():
            return True, 'bbob_tolupsigma (標準偏差が大きすぎるため停止)'
        if self.check_stagnation():
            return True, 'bbob_stagnation (評価値の進展が見られないため停止)'
        if self.check_conditioncov():
            return True, 'bbob_conditioncov (共分散行列の条件数が大きすぎるため停止)'
        if self.check_noeffectaxis():
            return True, 'bbob_noeffectaxis (軸方向で効果がなくなっているため停止)'
        if self.check_noeffectcoor():
            return True, 'bbob_noeffectcoor (座標方向で効果がなくなっているため停止)'
        if self.check_flat():
            return True, 'bbob_flat (評価値がすべて同じため停止)'
        # ここで新しい終了条件をチェック
        if self.check_no_improvement_generations():
            return True, ('no_improvement_generations (最小評価値の更新が、'
                          f'{self._cma.N * 200} 世代連続で見られなかったため停止)')
        return False, ''
    
# DdCmaの実行過程を記録し、後で解析できるようにするためのロガー
class Logger:
    """Logger for dd-CMA"""
    def __init__(self, cma, prefix='log', variable_list=['xmean', 'D', 'S', 'sigma', 'beta']):
        """
        Parameters
        ----------
        cma : DdCma instance
        prefix : string
            prefix for the log file path
        variable_list : list of string
            list of names of attributes of `cma` to be monitored
        """
        self._cma = cma
        self.neval_offset = 0
        self.t_offset = 0
        self.prefix = prefix
        self.variable_list = variable_list
        self.logger = dict()
        # 最小評価値を記録するファイル（.dat:一般的なデータファイルの拡張子で、特定の形式や内容に縛られずに使用できる汎用的なファイル)
        self.fmin_logger = self.prefix + '_fmin.dat'
        with open(self.fmin_logger, 'w') as f:
            f.write('#' + type(self).__name__ + "\n")
        # variable_listの中に入っているkeyについても記録する処理
        for key in self.variable_list:
            self.logger[key] = self.prefix + '_' + key + '.dat'
            with open(self.logger[key], 'w') as f:
                f.write('#' + type(self).__name__ + "\n")
    
    # Loggerのインスタンスを関数のように呼び出せるようにする（logger() で log() が実行される）
    def __call__(self, condition=''):
        self.log(condition)
    
    # CMAをリスタート時に更新
    def setcma(self, cma):
        # 累積評価回数(neval_offset)と世代数(t_offset)を累積していくために更新
        self.neval_offset += self._cma.neval
        self.t_offset += self._cma.t
        # リスタート時に設定したDdCmaに更新する
        self._cma = cma

    # ログをファイルに書き込む (世代数, 評価関数が呼び出された回数, 対象とするkeyの値の順)
    def log(self, condition=''):
        neval = self.neval_offset + self._cma.neval
        t = self.t_offset + self._cma.t
        # log_fmin.datに世代数(t), 評価回数(neval), 最小評価値(np.min(self._cma.arf))を記録
        with open(self.fmin_logger, 'a') as f:
            f.write("{} {} {}\n".format(t, neval, np.min(self._cma.arf)))
            # 終了条件(condition)があれば書き込む
            if condition:
                f.write('# End with condition = ' + condition)
        # variable_listの中に入っているkeyについても同様に記録する。終了条件は書き込まない。
        for key, log in self.logger.items():
            key_split = key.split('.')
            key = key_split.pop(0)
            var = getattr(self._cma, key)  
            for i in key_split:
                var = getattr(var, i)  
            if isinstance(var, np.ndarray) and len(var.shape) > 1:
                var = var.flatten()
            varlist = np.hstack((t, neval, var))
            with open(log, 'a') as f:
                f.write(' '.join(map(str, varlist)) + "\n")  # str に変更


    def my_formatter(self, x, pos):
        """Float Number Format for Axes"""
        float_str = "{0:2.1e}".format(x)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"{0}e{1}".format(base, int(exponent))
        else:
            return r"" + float_str + ""
        
    def plot(self,
             xaxis=0,
             ncols=None,
             figsize=None,
             cmap_='Spectral'):
        
        """Plot the result
        Parameters
        ----------
        xaxis : int, optional (default = 0)
            0. vs iterations
            1. vs function evaluations
        ncols : int, optional (default = None)
            number of columns
        figsize : tuple, optional (default = None)
            figure size
        cmap_ : string, optional (default = 'spectral')
            cmap
        
        Returns
        -------
        fig : figure object.
            figure object
        axdict : dictionary of axes
            the keys are the names of variables given in `variable_list`
        """
        mpl.rc('lines', linewidth=2, markersize=8)
        mpl.rc('font', size=12)
        mpl.rc('grid', color='0.75', linestyle=':')
        mpl.rc('ps', useafm=True)  # Force to use
        mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
        mpl.rc('text', usetex=False)  # for a paper submision

        prefix = self.prefix
        variable_list = self.variable_list

        # Default settings
        nfigs = 1 + len(variable_list)
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(nfigs)))
        nrows = int(np.ceil(nfigs / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)
        axdict = dict()
        
        # Figure
        fig = plt.figure(figsize=figsize)
        # The first figure
        x = np.loadtxt(prefix + '_fmin.dat')
        x = x[~np.isnan(x[:, xaxis]), :]  # remove columns where xaxis is nan
        # Axis
        ax = plt.subplot(nrows, ncols, 1)
        ax.set_title('fmin')
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        plt.plot(x[:, xaxis], x[:, 2:])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.my_formatter))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.my_formatter))
        axdict['fmin'] = ax

        # The other figures
        idx = 1
        for key in variable_list:
            idx += 1
            x = np.loadtxt(prefix + '_' + key + '.dat')
            x = x[~np.isnan(
                x[:, xaxis]), :]  # remove columns where xaxis is nan
            ax = plt.subplot(nrows, ncols, idx)
            # ax.set_title(r'\detokenize{' + key + '}')
            ax.set_title(key)
            ax.grid(True)
            ax.grid(which='major', linewidth=0.50)
            ax.grid(which='minor', linewidth=0.25)
            cmap = plt.get_cmap(cmap_)
            cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
            for i in range(x.shape[1] - 2):
                plt.plot(
                    x[:, xaxis], x[:, 2 + i], color=scalarMap.to_rgba(i))
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self.my_formatter))
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self.my_formatter))
            axdict[key] = ax

        plt.tight_layout() # not sure if it works fine
        return fig, axdict

"""
# この関数は使われていない
def random_rotation(self, func, dim):
    R = np.random.normal(0, 1, (dim, dim))
    for i in range(dim):
        for j in range(i):
            R[:, i] = R[:, i] - np.dot(R[:, i], R[:, j]) * R[:, j]
        R[:, i] = R[:, i] / np.linalg.norm(R[:, i])
    def rotatedfunc(x):
        return func(np.dot(x, R.T))
    return rotatedfunc
"""

# 境界制約や周期制約を処理するための関数で、特定の範囲外に出た解を反射的に戻したり、周期的な変数を指定範囲内に再配置する
# ここを各探索範囲ごとに制限するように変更する#######
# def mirror(z, lbound, ubound, flg_periodic):
#     """Mirroring Box-Constraint Handling and Periodic Constraint Handling
#     Parameters
#     ----------
#     z : ndarray (1D or 2D)
#         solutions to be corrected (修正対象の解)
#     lbound, ubound : ndarray (1D)
#         lower and upper bounds (各変数の下限・上限)
#         If some variables are not bounded, set np.inf or -np.inf
#     flg_periodic : ndarray (1D, bool)
#         flag for periodic variables (各変数が周期的かどうかを示すフラグ)
            
#     Returns
#     -------
#     projected solution in [lbound, ubound]
#     """
#     zz = np.copy(z)
#     flg_lower = np.isfinite(lbound) * np.logical_not(np.isfinite(ubound) + flg_periodic)
#     flg_upper = np.isfinite(ubound) * np.logical_not(np.isfinite(lbound) + flg_periodic)
#     flg_box = np.isfinite(lbound) * np.isfinite(ubound) * np.logical_not(flg_periodic)
#     width = ubound - lbound
#     if zz.ndim == 1:
#         zz[flg_periodic] = lbound[flg_periodic] + np.mod(zz[flg_periodic] - lbound[flg_periodic], width[flg_periodic])
#         zz[flg_lower] = lbound[flg_lower] + np.abs(zz[flg_lower] - lbound[flg_lower])
#         zz[flg_upper] = ubound[flg_upper] - np.abs(zz[flg_upper] - ubound[flg_upper])  
#         zz[flg_box] = ubound[flg_box] - np.abs(np.mod(zz[flg_box] - lbound[flg_box], 2 * width[flg_box]) - width[flg_box])
#     elif zz.ndim == 2:
#         zz[:, flg_periodic] = lbound[flg_periodic] + np.mod(zz[:, flg_periodic] - lbound[flg_periodic], width[flg_periodic])
#         zz[:, flg_lower] = lbound[flg_lower] + np.abs(zz[:, flg_lower] - lbound[flg_lower])
#         zz[:, flg_upper] = ubound[flg_upper] - np.abs(zz[:, flg_upper] - ubound[flg_upper])
#         zz[:, flg_box] = ubound[flg_box] - np.abs(np.mod(zz[:, flg_box] - lbound[flg_box], 2 * width[flg_box]) - width[flg_box])     
#     return zz



# if __name__ == "__main__":

#     # Ellipsoid-Cigar function
#     N = 10
    
#     #ここの関数を自分のものに変更する
#     # ここでのxは'array of x'でありlan行N列である
#     def ellcig(x):
#         cig = np.ones(x.shape[1]) / np.sqrt(x.shape[1])
#         d = np.logspace(0, 3, base=10, num=x.shape[1], endpoint=True)
#         y = x * d
#         f = 1e4 * np.sum(y ** 2, axis=1) + (1. - 1e4) * np.dot(y, cig)**2
#         return f
    
#     # Support for box constraint and periodic variables
#     # Set np.nan, -np.inf or np.inf if no bound
#     # ここを各探索範囲ごとに制限するように変更する#######
#     LOWER_BOUND = -5.0 * np.ones(N)
#     UPPER_BOUND = 5.0 * np.ones(N)
#     FLAG_PERIODIC = np.asarray([False] * N) # FLAG_PERIODICで変数が周期的であるかをフラグで示している（今回はすべて非周期的）
#     period_length = (UPPER_BOUND - LOWER_BOUND) * 2.0
#     period_length[FLAG_PERIODIC] /= 2.0
#     period_length[np.logical_not(np.isfinite(period_length))] = np.inf
    
#     # fobjは、ellcig関数に制約と周期性を適用した関数で、最適化アルゴリズムで実際に評価される目的関数
#     # mirrorで使えるデータにした後ellocogに与える
#     def fobj(x):
#         xx = mirror(x, LOWER_BOUND, UPPER_BOUND, FLAG_PERIODIC)
#         return ellcig(xx)
    
#     # Setting for resart
#     NUM_RESTART = 10  # number of restarts with increased population size
#     MAX_NEVAL = 1e6   # maximal number of f-calls
#     F_TARGET = 1e-8   # target function value(#############ここは自分で設定するべき#################)
#     total_neval = 0   # total number of f-calls(number of evaluation)
    
#     # Main loop
#     ddcma = DdCma(xmean0=np.random.randn(N), sigma0=np.ones(N)*2.) ####平均0、標準偏差はN次元の1の配列に2をかけている。他はデフォ値
#     ddcma.upper_bounding_coordinate_std(period_length)
#     checker = Checker(ddcma)
#     logger = Logger(ddcma)
#     for restart in range(NUM_RESTART):        
#         issatisfied = False
#         fbestsofar = np.inf
#         while not issatisfied:
#             ddcma.onestep(func=fobj)
#             ddcma.upper_bounding_coordinate_std(period_length)
#             fbest = np.min(ddcma.arf)
#             fbestsofar = min(fbest, fbestsofar)
#             #各ステップで最小の評価値fbestが目標値F_TARGETに到達したか、またはCheckerで停止条件を確認
#             # issatifiedにTrueかFalseを代入し、conditionに'終了条件内容'か'空白'を代入する(checkerのbbob_check参照)
#             if fbest <= F_TARGET:
#                 issatisfied, condition = True, 'ftarget'
#             else:
#                 issatisfied, condition = checker()
#             # 指定の間隔で現在の進行状況が出力
#             if ddcma.t % 10 == 0:
#                 print(ddcma.t, ddcma.neval, fbest, fbestsofar)
#                 logger()
#         # Loggerで記録が行われる
#         logger(condition)
#         print("Terminated with condition: " + str(condition))
#         # For restart
#         total_neval += ddcma.neval
#         # 合計の評価回数がMAX_NEVAL(最大評価回数)未満で目標値に到達していない場合、集団サイズを2倍にしてrestart
#         # そもそも何回も再スタートする意味があるかどうかは微妙
#         if total_neval < MAX_NEVAL and fbest > F_TARGET:
#             popsize = ddcma.lam * 2
#             ddcma = DdCma(xmean0=np.random.randn(N), sigma0=np.ones(N)*2., lam=popsize)
#             checker = Checker(ddcma)
#             logger.setcma(ddcma)
#             print("Restart with popsize: " + str(ddcma.lam))
#         # 終了条件を満たしているときは'for restart'を抜ける
#         else:
#             break

#     # Produce a figure
#     fig, axdict = logger.plot()
#     for key in axdict:
#         if key not in ('xmean'):
#             axdict[key].set_yscale('log')
#     plt.tight_layout()
#     plt.savefig(logger.prefix + '.pdf')
#     """
#     Contents of Figure
#     ------------------
#     fmin  xmean  D
#     S     sigma  beta
#     ------------------
#     fmin : Minimum evaluation value
#     xmean : Mean of x (x is an array containing variables)
#     D : Deviation
#     S : Standat deviation
#     sigma : Parameters for the scale of the search range of the optimization
#     beta : Parameters that control the frequency and intensity of matrix updates
#     """