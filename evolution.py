from magnus import MagnusIntegrator, get_example, braket
import numpy as np
import matplotlib.pyplot as plt


T = 1.
tau = 0.01
id = 'rz'

A, y0 = get_example(id)

mi = MagnusIntegrator(order=2)

ts, ys = mi.evolve(A, y0, T, tau)

y_test = np.array([1., -1.]) / np.sqrt(2.)
y_test = np.array([0., 1.])
tp = np.abs(mi.t_braket(y_test, ys))

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)

ax_1.plot(ts, np.real(ys[:, 0]), '--', label=r'$Re\{Psi_0\}$')
ax_1.plot(ts, np.real(ys[:, 1]), ':', label=r'$Re\{Psi_1\}$')

ax_1.plot(ts, tp, '-', label=r'$T$')
#ax_1.set_xlabel(r'$t$')
ax_1.legend()

ax_2.plot(ts, np.real(braket(ys, ys)))
ax_2.set_xlabel(r'$t$')
ax_2.set_ylabel(r'$\mid\Psi\mid^2$')

plt.show()
