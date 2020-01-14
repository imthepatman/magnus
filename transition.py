from magnus import MagnusIntegrator, get_example, braket, magnus_ana
import numpy as np
import matplotlib.pyplot as plt

order_max = 4
qf = 'simpson'

example_id = 'rect'

T = 1.
tau = 0.1

om_max = 15.
N_om = 100
V0 = np.pi/2.
V0 = 2. # ~gamma in paper



mis = [MagnusIntegrator(order=i + 1, qf=qf) for i in range(order_max)]

oms = np.linspace(0., om_max, N_om)    # ~ xi in paper
tp_oms = []
tp_oms_ana = []
tp_oms_exact = []

y_trans = np.array([1., 0.])

for i, om in enumerate(oms):
    A, y0, p_ex, pm1 = get_example(example_id, om=om, V0=V0)

    tps = []
    for mi in mis:
        y_next = mi.evolve(A, y0, T, tau=tau)
        # computing transition probability
        tp = np.abs(np.dot(y_trans, y_next))**2
        tps.append(tp)

    tp_oms_exact.append(p_ex(T))
    tp_oms_ana.append(pm1(T))
    tp_oms.append(tps)

    if (i+1)%(len(oms)//10) == 0:
        print('om {:d}/{:d}: {:0.2f} done'.format(i + 1, len(oms), om))

fig = plt.figure(figsize=(9, 6))
ax_1 = fig.add_subplot(111)

ax_1.plot(oms, tp_oms_exact, '-', label=r'exact')
ax_1.plot(oms, tp_oms_ana, ':', label=r'$M_{ana} - 1$')

for i, tp in enumerate(np.transpose(tp_oms)):
    ax_1.plot(oms, tp, '--', label=r'$M-{}$'.format(i + 1))

ax_1.set_xlabel(r'$\omega$')
ax_1.set_ylabel(r'$P_T$')
ax_1.legend()

plt.show()
