from magnus import MagnusIntegrator, get_example, braket, sig_x
import numpy as np
import matplotlib.pyplot as plt

T = 1.
Nt = 20
id = 'rect'
V0 = 2.
om = np.pi

order = 4
qf = 'simpson'

A, y0, _, _ = get_example(id, om=om, V0=V0)

mi = MagnusIntegrator(order=order, qf=qf)

ts = np.linspace(0., T, Nt, endpoint=True)

y_forward = np.zeros((Nt, len(y0)), dtype=np.complex)
y_forward[0] = y0
for i in range(1, Nt):
    y_forward[i] = mi.evolve(A, y0=y_forward[i - 1], t0=ts[i - 1], T=ts[i], n_subint=1)
y_final = mi.evolve(A, y0=y0, T=T, n_subint=Nt)

y_backward = np.zeros((Nt, len(y0)), dtype=np.complex)
y_backward[0] = y_forward[-1]
for i in range(1, Nt):
    y_backward[i] = mi.evolve(A, y0=y_backward[i - 1], t0=ts[- i], T=ts[- i - 1], n_subint=1)

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)

ax_1.plot(ts, np.real(y_forward[:, 0]), '--', label=r'$Re\{Psi_0\}$')
ax_1.plot(ts, np.real(y_forward[:, 1]), ':', label=r'$Re\{Psi_1\}$')
ax_1.plot(T, np.real(y_final[0]), 'ro', label=r'$Re\{Psi_{fin,0}\}$')
ax_1.plot(T, np.real(y_final[1]), 'go', label=r'$Re\{Psi_{fin,1}\}$')

ax_1.plot(np.flip(ts), np.real(y_backward[:, 0]), ':', label=r'$Re\{\Phi_0\}$')
ax_1.plot(np.flip(ts), np.real(y_backward[:, 1]), '--', label=r'$Re\{\Phi_1\}$')

ax_1.legend()

ax_2.plot(ts, np.real(braket(y_forward, y_forward)))
ax_2.set_xlabel(r'$t$')
ax_2.set_ylabel(r'$\mid\Psi\mid^2$')

plt.show()
