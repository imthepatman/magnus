from magnus import MagnusIntegrator
import numpy as np
import matplotlib.pyplot as plt

A = lambda t: np.array([[1j, 2 + 1j - np.cos(2 * np.pi * t)], [-2 + 1j + np.cos(2 * np.pi * t), 3 * 1j]])

#A = lambda t: 1j*np.array([[1., 0.], [0., -1.]])
y0 = np.array([1.,1.]) / np.sqrt(2.)
#y0 = np.array([1., 0.])

T = 10.
tau = 0.1

mi = MagnusIntegrator(A)

ts, ys = mi.evolve(y0, T, tau)

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)

ax_1.plot(ts, np.real(ys[:, 0]), '-', ts, np.real(ys[:, 1]), '-')
#ax_1.set_xlabel(r'$t$')
ax_1.set_ylabel(r'$Re\{\Psi\}$')

ax_2.plot(ts, mi.norm(ys))
ax_2.set_xlabel(r'$t$')
ax_2.set_ylabel(r'$\mid\Psi\mid}$')

plt.show()
