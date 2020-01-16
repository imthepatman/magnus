import numpy as np
import scipy.integrate as integrate
from scipy.linalg import expm
from scipy.special import bernoulli, factorial, roots_legendre
import itertools as it

sig_x = np.array([[0., 1.], [1., 0.]])
sig_y = np.array([[0., -1.j], [1.j, 0.]])
sig_z = np.array([[1., 0.], [0., -1.]])


def get_example(id='std', **kwargs):
    if id == 'std':
        H = lambda t: 1.j * np.array([[1j, 2 + 1j - np.cos(2 * np.pi * t)], [-2 + 1j + np.cos(2 * np.pi * t), 3 * 1j]])
        y0 = np.array([1., 1.]) / np.sqrt(2.)
    elif id == 'rect':
        om = kwargs.get('om', 1.)
        V0 = kwargs.get('V0', 1.)

        V = lambda t: V0
        H = lambda t: V(t) * (sig_x * np.cos(om * t) - sig_y * np.sin(om * t))
        y0 = np.array([0., 1.])
        p_ex = lambda t: 4. / (4. + (om / V0) ** 2) * np.sin(np.sqrt(V0 ** 2 + om ** 2 / 4.) * t) ** 2
        pm1 = lambda t: np.sin(2. * V0 / om * np.sin(om / 2. * t)) ** 2 if om != 0. else np.sin(V0) ** 2

    elif id == 'rz':
        om = kwargs.get('om', 1.)
        V0 = kwargs.get('V0', 1.5)

        V = lambda t: V0 / np.cosh(t/np.pi)
        H = lambda t: V(t) * (sig_x * np.cos(np.pi * om * t) - sig_y * np.sin(np.pi * om * t))
        y0 = np.array([0., 1.])
        p_ex = lambda t: np.sin(V0 * t) ** 2 / np.cosh(np.pi * om * t / 2.) ** 2
        pm1 = lambda t: np.sin(V0 * t / np.cosh(np.pi * om * t / 2.)) ** 2

    A = lambda t: -1.j * H(t)
    return A, y0, p_ex, pm1


def magnus_ana(t, om, V0):
    Om_1 = -1j * V0 / om * (sig_x * np.sin(om * t) + sig_y * (1. - np.cos(om * t)))

    Om = Om_1
    U = expm(Om)
    return U, Om


class MagnusIntegrator:
    def __init__(self, order=2, qf='midpoint', explicit_low=False, mute=True):
        # to have gauss Legendre of order n, type qf = 'gln', e.g. 'gl4'
        self.order = order
        self.qf = qf
        self.explicit_low = explicit_low
        self.mute = mute

        b_n = bernoulli(self.order - 1)
        fac_n = np.array([factorial(n) for n in range(self.order)])
        self.combfacs = b_n / fac_n

        self.ks = {}
        for n in range(2, self.order + 1):
            for j in range(1, n):
                self.ks['{}_{}'.format(n, j)] = self.k_combs(n, j)

        if not self.mute:
            print('combinatorial factors and k indices combinations')
            for n in range(1, self.order + 1):
                print('n', n)
                for j in range(1, n):
                    print('j', j)
                    print('cf', self.combfacs[j], self.ks['{}_{}'.format(n, j)])
                print()

    def Omega(self):
        Om = self.fOmega(self.T, n=1)
        for n in range(2, self.order + 1):
            Om_n = self.fOmega(self.T, n=n)
            Om += Om_n

        return Om

    def evolve(self, A, y0, T, tau=0.01, t0=0.):
        self.A = A
        self.t0 = t0
        self.T = T
        self.tau = tau

        y_next = expm(self.Omega()) @ y0
        return y_next

    ### Auxiliary functions ###

    def fOmega(self, s, n=1):
        if n == 1:
            Om_1 = self.quad(self.A, s)
            return Om_1
        elif not self.explicit_low or n > 3:
            Om_n = self.quad(self.fOmega_p, s, n=n)
            return Om_n
        else:
            if n == 2:
                Om_2 = 0.5 * self.quad(lambda t1: self.comm(self.A(t1), self.quad(self.A, t1)), s)
                return Om_2
            elif n == 3:
                omp_3 = lambda t1, t2, t3: self.comm(self.A(t1), self.comm(self.A(t2), self.A(t3))) + self.comm(self.comm(self.A(t1), self.A(t2)), self.A(t3))
                Om_3 = 1. / 6. * self.quad(lambda t1: self.quad(lambda t2: self.quad(lambda t3: omp_3(t1, t2, t3), t2), t1), s)
                return Om_3

    def fOmega_p(self, s, n=2):
        omp = 0.
        for j in range(1, n):
            if self.combfacs[j] != 0.:
                S_j = 0.
                # loop over possible variations for given j
                for k in self.ks['{}_{}'.format(n, j)]:
                    # loop through indices in variation
                    # right sided, nested commutators
                    s_k = self.comm(self.fOmega(s, n=k[0]), self.A(s))
                    for ik in range(1, len(k)):
                        s_k = self.comm(self.fOmega(s, n=k[ik]), s_k)
                    S_j += s_k
                # sum over j before integration
                omp += self.combfacs[j] * S_j
        return omp

    def k_combs(self, n, j):
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
        c = M1 @ M2 - M2 @ M1
        return c

    def quad(self, f, T, **kwargs):

        Nt = int((T - self.t0) / self.tau)
        ts = np.linspace(self.t0, T, Nt, endpoint=True)

        res = np.zeros_like(self.A(0.))
        for i in range(1, len(ts)):
            t_start = ts[i - 1]
            t_end = ts[i]
            t_mid = (t_end + t_start) / 2.
            dt = t_end - t_start

            if self.qf == 'midpoint':
                res += f(t_start + dt / 2., **kwargs) * dt

            elif self.qf == 'simpson':
                res += dt / 6. * (f(t_start, **kwargs) + 4. * f(t_mid, **kwargs) + f(t_end, **kwargs))
            elif self.qf == 'gauss2':
                xi = [-1. / np.sqrt(3.), 1. / np.sqrt(3.)]
                res += dt / 2. * (f(t_mid + dt / 2. * xi[0], **kwargs) + f(t_mid + dt / 2. * xi[1], **kwargs))

            elif self.qf[0:2] == 'gl':  # gauss legendre
                n = int(self.qf[-1])  # take the order of Gauss Legendre quadr. formula
                x, w = roots_legendre(n)
                for i in range(0, n):
                    res += w[i] * f(((t_end - t_start) / 2) * x[i] + (t_end + t_start) / 2, **kwargs) * (t_end - t_start) / 2

        return res

    def t_braket(self, phi, psi_t):
        return np.einsum('i, ni->n', np.conj(phi), psi_t)


def braket(phi, psi):
    return np.sum(np.conj(phi) * psi, axis=-1)
