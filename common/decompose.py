import time

import numpy as np
from common.logger import get_logger

logger = get_logger(__name__)
EPS = 1e-35


class NMFD:
    def __init__(self, v, k, dt, init_w=None, init_h=None, n_iter=30, metric='KL'):
        """
        Non-negative Matrix Factor Deconvolution class
        Parameters
        ----------
        v: observed signal (input) with time-frequency domain (np.array(F x T))
        k: number of spectral bases
        dt: time length of spectral bases
        init_w: init matrix of spectral bases (np.array(F x k x dt))
        init_h: init matrix of activation (np.array(k x T))
        n_iter: iteration number of NMFD
        metric: metric between observed signal and model
        """
        np.random.seed(0)

        freq_num, time_num = v.shape[0], v.shape[1]
        self.v = v
        self.k = k
        self.dt = dt

        self.w = np.random.random(size=(freq_num, k, dt))
        if init_w is not None:
            assert init_w.shape == self.w.shape, f'init_w size error. correct size: {self.w.shape}'
            self.w = init_w

        self.h = np.random.random(size=(k, time_num))
        if init_h is not None:
            assert init_h.shape == self.h.shape, f'init_h size error. correct size: {self.h.shape}'
            self.h = init_h

        self.n_iter = n_iter

        assert metric in ['KL', 'euclid'], f'metric type error. input "KL" or "euclid" for now.'
        self.metric = metric

        self.obj_history = []

    def get_wh(self):
        res = np.zeros(self.v.shape)
        for _dt in range(self.dt):
            res += self.w[:, :, _dt] @ self.shift(self.h, _dt)
        return res + EPS

    @staticmethod
    def shift(org: np.array, s: int):
        assert org.shape[1] > s, f'Input shift_num less than {org.shape[1]}'
        shift = np.zeros(org.shape)
        if s > 0:
            shift[:, s:] = org[:, :-s]
        elif s < 0:
            shift[:, :s] = org[:, -s:]
        else:
            shift = org
        return shift

    def normalize_w(self):
        for _k in range(self.k):
            self.w[:, _k, :] = self.w[:, _k, :] / self.w[:, _k, :].sum()
        return

    def update_w(self):
        w = np.zeros(self.w.shape) + self.w
        for _dt in range(self.dt):
            numerator = (self.v / self.get_wh()) @ self.shift(self.h, _dt).T
            denominator = np.ones(self.v.shape) @ self.shift(self.h, _dt).T
            w[:, :, _dt] = w[:, :, _dt] * (numerator / denominator)
        self.w = w
        return

    def update_h(self):
        numerator = np.zeros(self.h.shape)
        denominator = np.zeros(self.h.shape)
        for _dt in range(self.dt):
            numerator += self.w[:, :, _dt].T @ self.shift(self.v / self.get_wh(), -_dt)
            denominator += self.w[:, :, _dt].T @ np.ones(self.v.shape)
        # print(f'[before] h size: {self.h.shape}')
        # print(f'numerator: {numerator.shape}')
        # print(f'denominator: {denominator.shape}')
        self.h = self.h * (numerator / denominator)
        # print(f'[after]  h size: {self.h.shape}')
        return

    def calc_objective_function(self):
        obj = self.v * np.log(self.v / self.get_wh()) - self.v + self.get_wh()
        return obj.sum()

    def fit(self):
        for n in range(self.n_iter):
            st = time.time()
            self.update_w()
            self.normalize_w()
            self.update_h()
            obj_value = self.calc_objective_function()
            et = time.time()
            logger.info(f'NMFD iteration {n} - objective_function {obj_value} - {et-st} sec.')
            self.obj_history.append(obj_value)

        return self.w, self.h

    def masking_ith_component(self, i, p=1.0):
        x = self.get_wh()

        mask = np.zeros(self.v.shape)
        for _dt in range(self.dt):
            mask += self.w[:, [i], _dt] @ self.shift(self.h[[i], :], _dt)

        return self.v * (mask / x) ** p
