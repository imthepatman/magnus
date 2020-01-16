from magnus import MagnusIntegrator, get_example, braket, sig_x
import numpy as np
import matplotlib.pyplot as plt

T = 40.
Nt = 100
id = 'free'

t_plot_skip = 10

order = 3
qf = 'gl4'

A, y0, xs, V = get_example(id, V0=10.)

mi = MagnusIntegrator(order=order, qf=qf)

ts = np.linspace(0., T, Nt + 1, endpoint=True)

y_forward = np.zeros((Nt, len(y0)), dtype=np.complex)
y_forward[0] = y0
for i in range(1, Nt):
    y_forward[i] = mi.evolve(A, y0=y_forward[i - 1], t0=ts[i - 1], T=ts[i], n_subint=1)

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(211)

for i in range(1, Nt):
    if i%t_plot_skip==0:
        ax_1.plot(xs, np.abs(y_forward[i]), '-', label=r'$Re\{Psi('+'{:0.2f}'.format(ts[i]) +')\}$')

ax_1.plot(xs, V(0.), ':', label=r'$V(0)$')

ax_1.legend()

plt.show()
