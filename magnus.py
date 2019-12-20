import numpy as np
import scipy.integrate as integrate
from scipy.linalg import expm
from scipy.special import bernoulli, factorial
import itertools as it

sig_x = np.array([[0., 1.], [1., 0.]])
sig_y = np.array([[0., -1.j], [1.j, 0.]])
sig_z = np.array([[1., 0.], [0., -1.]])

def get_example(id='std'):
    if id == 'std':
        A = lambda t: np.array([[1j, 2 + 1j - np.cos(2 * np.pi * t)], [-2 + 1j + np.cos(2 * np.pi * t), 3 * 1j]])
        y0 = np.array([1., 1.]) / np.sqrt(2.)
    elif id == 'rect':
        f = lambda t: 1. if t>=0. else 0.
        A = lambda t: -1.j*f(t)*(sig_x*np.cos(t) - sig_y*np.sin(t))
        y0 = np.array([1., 1.]) / np.sqrt(2.)
        #y0 = np.array([1., 0.])

    return A, y0


class MagnusIntegrator:
    def __init__(self, A, order=2, qf='midpoint'):
        self.A = A
        self.order = order
        self.qf = qf

        self.b_n = bernoulli(self.order)
        self.fac_n = np.array([factorial(n) for n in range(self.order + 1)])

        self.ks = {}
        for n in range(2, self.order + 1):
            for j in range(1, n):
                self.ks['{}_{}'.format(n, j)] = self.gen_ks(n, j)

    def Omega(self, t_s, t_e):
        self.t0 = t_s
        Om = self.fOmega(t_e, n=1)
        for n in range(2, self.order + 1):
            Om_n = self.fOmega(t_e, n=n)
            Om += Om_n

        return Om

    def evolve(self, y0, T, tau=0.01):
        Nt = int(T / tau)
        ts = np.arange(0, Nt + 1) * tau
        if Nt * tau < T:
            ts = np.append(ts, T)

        ys = np.zeros((len(ts), len(y0)), dtype=np.complex)
        ys[0] = y0
        for n in range(1, len(ts)):
            ys[n] = expm(self.Omega(ts[n - 1], ts[n])) @ ys[n - 1]

        return ts, ys

    ### Auxiliary functions ###

    def fOmega(self, s, n=1):
        if n < 2:
            Om_1 = self.quad(self.A, s)
            return Om_1
        else:
            Om_n = self.quad(self.fOmega_p, s, n=n)
            return Om_n

    def fOmega_p(self, s, n=2):
        omp = 0.
        for j in range(1, n):
            k_js = self.gen_ks(n, j)
            S_j = 0.
            # loop over possible variations for given j
            for k in k_js:
                # right sided, nested commutators
                B_k = self.comm(self.fOmega(s, n=k[0]), self.A(s))
                # loop through indices in variation
                for ik in range(1, len(k)):
                    B_k += self.comm(self.fOmega(s, n=k[ik]), B_k)

                S_j += B_k
            # sum over j before integration
            omp += self.b_n[j] / self.fac_n[j] * S_j
        return omp

    def gen_ks(self, n, j):
        # generate possible variations of k which sum to n - 1
        k_range = np.arange(1, n - j + 1)
        k_var = np.array(list(it.product(k_range, repeat=j)))
        if len(k_var) > 0:
            if j < 2:
                k_var = k_var.reshape((len(k_var), 1))
            k_sum = np.sum(k_var, axis=1)

            mask = k_sum == n - 1
        k_js = k_var[mask]
        return k_js

    def comm(self, M1, M2):
        return M1 @ M2 - M2 @ M1

    def quad(self, f, t, **kwargs):
        if self.qf == 'midpoint':
            dt = t - self.t0
            res = f(self.t0 + dt / 2., **kwargs) * dt

            return res
        elif self.qf == 'simpson':
            #TODO to implement

            return None

    def t_braket(self, phi, psi_t):
        return np.einsum('i, ni->n', np.conj(phi), psi_t)

def braket(phi, psi):
    return np.sum(np.conj(phi) * psi, axis=-1)
