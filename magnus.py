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
        H = lambda t: 1.j*np.array([[1j, 2 + 1j - np.cos(2 * np.pi * t)], [-2 + 1j + np.cos(2 * np.pi * t), 3 * 1j]])
        y0 = np.array([1., 1.]) / np.sqrt(2.)
    elif id == 'rect':
        om = kwargs.get('om', 1.)
        V0 = kwargs.get('V0', 1.)

        f = lambda t: 1. if t>=0. else 0.
        H = lambda t: 0.5*om*sig_z + V0*f(t)*sig_x
        y0 = np.array([0., 1.])

    elif id=='rz':
        V0 = 1./np.pi
        T = 1.
        V = lambda t: V0/np.cosh(t)
        H = lambda t: V(t)*(sig_x * np.cos(t) - sig_y*np.sin(t))
        y0 = np.array([1., 0.])

    A = lambda t: -1.j*H(t)
    return A, y0


class MagnusIntegrator:
    def __init__(self, order=2, qf='simpson'):
        #to have gauss Legendre of order n, type qf = 'gln', e.g. 'gl4'
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
            print('n', n, 'Om', Om_n)
            Om += Om_n

        return Om

    def evolve(self, A, y0, T, tau=0.01):
        self.A = A
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
            #Om_n = 0.
            #for j in range(1, n):
            #    Om_n+= self.b_n[j] / self.fac_n[j]*self.quad(self.S, s, n=n, j=j)
            return Om_n

    def fOmega_p(self, s, n=2):
        omp = 0.
        for j in range(1, n):
            S_j = 0.
            # loop over possible variations for given j
            for k in self.ks['{}_{}'.format(n, j)]:
                # right sided, nested commutators
                B_k = self.comm(self.fOmega(s, n=k[0]), self.A(s))
                # loop through indices in variation
                for ik in range(1, len(k)):
                    B_k += self.comm(self.fOmega(s, n=k[ik]), B_k)

                S_j += B_k
            # sum over j before integration
            omp += self.b_n[j] / self.fac_n[j] * S_j
        return omp

    def S(self, s, n, j):
        S_j = 0.
        # loop over possible variations for given j
        for k in self.ks['{}_{}'.format(n, j)]:
            # right sided, nested commutators
            B_k = self.comm(self.fOmega(s, n=k[0]), self.A(s))
            # loop through indices in variation
            for ik in range(1, len(k)):
                B_k += self.comm(self.fOmega(s, n=k[ik]), B_k)

            S_j += B_k
        return S_j

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
        dt = t - self.t0
        if self.qf == 'midpoint':
            dt = t - self.t0
            res = f(self.t0 + dt / 2., **kwargs) * dt

        elif self.qf == 'simpson':
            tm = (self.t0 + t)/2.
            res = dt/6. * (f(self.t0, **kwargs) + 4. * f(tm, **kwargs) + f(t, **kwargs))

        elif self.qf[0:2] == 'gl': #gauss legendre
            a = self.t0
            b = t
            res = 0
            n = int(self.qf[-1]) #take the order of Gauss Legendre quadr. formula
            x,w = roots_legendre(n) 
            for i in range(0,n):
                res +=  w[i] * f( ((b-a)/2)*x[i] + (b+a)/2, **kwargs)
            res = res*(b-a)/2

        return res
    
    def t_braket(self, phi, psi_t):
        return np.einsum('i, ni->n', np.conj(phi), psi_t)

def braket(phi, psi):
    return np.sum(np.conj(phi) * psi, axis=-1)
