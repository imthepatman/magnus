import numpy as np
import scipy.integrate as integrate
from scipy.linalg import expm


class MagnusIntegrator:
    def __init__(self, A, order=1, N_int=10, qf = 'midpoint'):
        self.A = A
        self.order = order
        self.N_int = N_int
        self.qf = qf

    def Omega(self, t):
        Om_1 = self.quadrature(self.A, 0, t)

        Om = Om_1
        return Om

    def evolve(self, y0, T, tau=0.01):
        Nt = int(T / tau)
        ts = np.arange(0, Nt + 1) * tau
        if Nt * tau < T:
            ts = np.append(ts, T)
        taus = np.diff(ts)

        ys = np.zeros((len(ts), len(y0)), dtype=np.complex)
        ys[0] = y0
        for n in range(1, len(ts)):
            ys[n] = expm(self.Omega(taus[n - 1])) @ ys[n - 1]

        return ts, ys

    ### Auxiliary functions ###

    def comm(self, t1, t2):
        return self.A(t1) @ self.A(t2) - self.A(t2) @ self.A(t1)

    def quadrature(self, f, t0, t):
        if self.qf == 'midpoint':
            dt = t - t0
            res = f(t0 + dt / 2.) * dt

            return res
        elif self.qf == 'simpson':
            ts = np.linspace(t0, t, self.N_int, endpoint=True)
            fs = np.array([f(ti) for ti in ts])
            res = integrate.simps(fs, ts, axis=0)

            return res

    def norm(self, y):
        return np.sqrt(np.real(np.sum(np.conj(y)*y, axis=-1)))


