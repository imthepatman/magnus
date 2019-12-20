import numpy as np
import scipy.integrate as integrate
from scipy.linalg import expm
from scipy.special import bernoulli, factorial
import itertools as it


class MagnusIntegrator:
    def __init__(self, A, order=5, N_int=10, qf='midpoint'):
        self.A = A
        self.order = order
        self.N_int = N_int
        self.qf = qf

        self.b_n = bernoulli(self.order)
        self.fac_n = np.array([factorial(n) for n in range(self.order + 1)])

        self.ks = [[self.gen_ks(n, j) for j in range(1, n)] for n in range(2, self.order + 1)]

        #self.fOmega = self.gen_fOmega()
        # quit()

    def Omega(self, t_s, t_e):
        self.t0 = t_s
        Om = self.Om(t_e, n=1)
        for n in range(2, self.order + 1):
            Om += self.Om(t_e, n=n)

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

    def comm(self, M1, M2):
        return M1 @ M2 - M2 @ M1

    def Om(self, s, n=1):
        if n < 2:
            Om_1 = self.quad(self.A, s)
            return Om_1
        else:
            Om_n = self.quad(self.Omega_p, s, n=n)
            return Om_n

    def Omega_p(self, s, n=2):
        omp = 0.
        for j in range(1, n):
            print('j', j)

            k_js = self.gen_ks(n, j)

            S_j = 0.

            # loop over possible variations for given j
            for k in k_js:
                # right sided, nested commutators

                B_k = self.comm(self.Om(s, n=k[0]), self.A(s))
                # loop through indices in variation
                for ik in range(1, len(k)):
                    B_k += self.comm(self.Om(s, n=k[ik]), B_k)

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

    def gen_fOmega(self, t):
        fdict = {}
        fdict['Om_1'] = lambda s: self.quad(self.A, s)

        b_n = bernoulli(self.order)
        fac_n = np.array([factorial(n) for n in range(self.order + 1)])

        for n in range(2, self.order + 1):
            fdict['C_{}_{}'.format(n, 0)] = lambda s: 0.
            print('\nn', n)
            for j in range(1, n):
                print('j', j)

                # generate possible variations of k which sum to n - 1
                k_range = np.arange(1, n - j + 1)
                k_var = np.array(list(it.product(k_range, repeat=j)))
                if len(k_var) > 0:
                    if j < 2:
                        k_var = k_var.reshape((len(k_var), 1))
                    k_sum = np.sum(k_var, axis=1)

                    mask = k_sum == n - 1
                k_js = k_var[mask]
                print(k_js)

                fdict['S_{}_{}_{}'.format(n, j, 0)] = lambda s: 0.

                # loop over possible variations for given j
                for jk, k in enumerate(k_js):
                    # right sided, nested commutators

                    fdict['B_{}_{}_{}_{}'.format(n, j, jk, 0)] = lambda s: self.comm(fdict['Om_{}'.format(k[0])](s), self.A(s))
                    # loop through indices in variation
                    for ik in range(1, len(k)):
                        fdict['B_{}_{}_{}_{}'.format(n, j, jk, ik)] = lambda s: self.comm(fdict['Om_{}'.format(k[ik])](s), fdict['B_{}_{}_{}_{}'.format(n, j, jk, ik - 1)](s))

                    fdict['S_{}_{}_{}'.format(n, j, jk + 1)] = lambda s: fdict['S_{}_{}_{}'.format(n, j, jk)](s) + fdict['B_{}_{}_{}_{}'.format(n, j, jk, len(k) - 1)](s)
                # sum over j before integration

                fdict['C_{}_{}'.format(n, j)] = lambda s: fdict['C_{}_{}'.format(n, j - 1)](s) + b_n[j] / fac_n[j] * fdict['S_{}_{}_{}'.format(n, j, len(k_js))](s)

            fdict['Om_{}'.format(n)] = lambda s: self.quad(fdict['C_{}_{}'.format(n, n - 1)], s)

        print(fdict.keys())

        self.t0 = 0.
        print(fdict['C_{}_{}'.format(2, 1)](0.))
        quit()

        return fdict

    def quad(self, f, t, **kwargs):
        t0 = kwargs.get('t0', self.t0)
        if t0 in kwargs:
            kwargs.pop('t0')

        if self.qf == 'midpoint':
            dt = t - t0
            res = f(t0 + dt / 2., **kwargs) * dt

            return res
        elif self.qf == 'simpson':
            ts = np.linspace(t0, t, self.N_int, endpoint=True)
            fs = np.array([f(ti) for ti in ts])
            res = integrate.simps(fs, ts, axis=0)

            return res

    def norm(self, y):
        return np.sqrt(np.real(np.sum(np.conj(y) * y, axis=-1)))
