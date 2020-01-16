from magnus import MagnusIntegrator, get_example, braket, sig_x
import numpy as np
import matplotlib.pyplot as plt


T = 1.
tau = 0.1
id = 'rect'
V0 = 2.
om = np.pi

order = 4
qf = 'simpson'

A, y0,_,_ = get_example(id, om=om, V0=V0)

mi = MagnusIntegrator(order=order, qf=qf)

Nt = int(T/tau)
ts = np.linspace(0., T, Nt, endpoint=True)

ys = np.zeros((Nt, len(y0)), dtype=np.complex)
ys[0] = y0
for i in range(1, Nt):
    ys[i] = mi.evolve(A, y0=ys[i-1], t0=ts[i-1], T=ts[i], tau=tau)

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(211)
ax_2 = fig.add_subplot(212)

ax_1.plot(ts, np.real(ys[:, 0]), '--', label=r'$Re\{Psi_0\}$')
ax_1.plot(ts, np.real(ys[:, 1]), ':', label=r'$Re\{Psi_1\}$')

ax_1.legend()

ax_2.plot(ts, np.real(braket(ys, ys)))
ax_2.set_xlabel(r'$t$')
ax_2.set_ylabel(r'$\mid\Psi\mid^2$')

plt.show()
