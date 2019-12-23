from magnus import MagnusIntegrator, get_example, braket
import numpy as np
import matplotlib.pyplot as plt


mi_1 = MagnusIntegrator(order=1)
mi_2 = MagnusIntegrator(order=2)
mi_3 = MagnusIntegrator(order=3)
mi_4 = MagnusIntegrator(order=4)

mis = [mi_1, mi_2, mi_3, mi_4]

T = 0.1
oms = np.linspace(0, 15, 200)
tp_oms = []
y_test = np.array([1., 0.])

for om in oms:
    A, y0 = get_example('rect', om=om/T, V0=2./T)
    tps = []
    for mi in mis:
        _, ys = mi.evolve(A, y0, T, T)
        tp = np.abs(braket(y_test, ys[-1]))
        tps.append(tp)
    tp_oms.append(tps)

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(111)

for i, tp in enumerate(np.transpose(tp_oms)):
    ax_1.plot(oms, tp, '-', label=r'$M-{}$'.format(i+1))

ax_1.set_xlabel(r'$\omega$')
ax_1.set_ylabel(r'$P_T$')
ax_1.legend()

plt.show()
