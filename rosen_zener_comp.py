import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

xi =  1.0 #0.3  # 1.0
#xi = int(input('put an xi:'))


sig_x = np.array([[0., 1.], [1., 0.]])
sig_y = np.array([[0., -1.j], [1.j, 0.]])
sig_z = np.array([[1., 0.], [0., -1.]])


def pex(gamma):
    return np.sin(gamma) ** 2 / np.cosh(np.pi * xi / 2) ** 2


def rhs(t, gamma, T):
    # T: characteristic time (?) #TODO: understand what actually is
    V0 = gamma / (np.pi * T)
    
    return -1j * V0 / (np.cosh(t / T)) * (sig_x * np.cos(xi * (t / T)) - sig_y * np.sin(xi * (t / T)))


ti = -25.0
tf = +25.0 #sufficiently large time interval
T = 1

gamma_int = 70  # n° of gamma intervals
gamma_range = np.linspace(0, 2 * np.pi, gamma_int)

h1 = 1
ts1 = int((tf - ti) / h1)  # number of time steps
t = np.linspace(ti, tf, ts1 + 1)
y0 = np.array([1.0, 0.0])
yE = np.zeros([2, ts1 + 1], dtype=np.complex_)
yE[:, 0] = y0
yM2 = np.zeros([2, ts1 + 1], dtype = np.complex_)
yM2[:,0] = y0
tpsE = []  # Euler transition probabilities
tpsRK = [] # Runge Kutta
tpsM2 = [] # Magnus 2
tpsM4 = [] # Magnus 4
y_trans = np.array([0.0, 1.0]) # final state (according to paper)

for gamma in gamma_range:
    for i in range(0, ts1):
        yE[:, i + 1] = yE[:, i] + h1 * rhs(t[i], gamma, T) @ yE[:, i]  # Euler iteration
        yM2[:, i+1] = expm(h1*rhs(t[i] + h1 /2 ,gamma,T))@yM2[:,i] #Magnus exponential midpoint
    
    tp = np.abs(np.dot(y_trans, yE[:, -1])) ** 2
    tpsE.append(tp)
    tp = np.abs(np.dot(y_trans, yM2[:, -1])) ** 2
    tpsM2.append(tp)

# integrators with different no of time steps: same number of function evaluations
h2 = 2
ts2 = int((tf - ti) / h2)  # number of time steps
t = np.linspace(ti, tf, ts2 + 1)
yRK = np.zeros([2, ts2 + 1], dtype=np.complex_)
yRK[:, 0] = y0
yM4 = np.zeros([2, ts2 + 1], dtype=np.complex_)
yM4[:, 0] = y0

c1 = 1/2 - np.sqrt(3)/6
c2 = 1/2 + np.sqrt(3)/6

def comm(A,B): 
    #commutator
    return A@B - B@A

    
for gamma in gamma_range:
    for i in range(0, ts2):
        # Runge Kutta Steps
        s1 = yRK[:, i]
        s2 = yRK[:, i] + (h2 / 2) * rhs(t[i], gamma, T) @ s1
        s3 = yRK[:, i] + (h2 / 2) * rhs(t[i] + h2 / 2, gamma, T) @ s2
        s4 = yRK[:, i] + h2 * rhs(t[i] + h2 / 2, gamma, T) @ s3
        yRK[:, i + 1] = yRK[:, i] + (h2 / 6) * (rhs(t[i], gamma, T) @ s1 + 2 * rhs(t[i] + (h2 / 2), gamma, T) @ s2 + 2 * rhs(t[i] + (h2 / 2), gamma, T) @ s3 + rhs(t[i] + h2, gamma,T) @ s4)
        
        yM4[:,i+1] = expm(0.5*h2*( rhs(t[i]+c1*h2,gamma,T) + rhs(t[i]+c2*h2, gamma, T) ) + (np.sqrt(3)/12)*(h2**2)*comm(rhs(t[i]+c2*h2, gamma, T),rhs(t[i]+c1*h2, gamma, T)))@yM4[:,i]
   
    tp = np.abs(np.dot(y_trans, yRK[:, -1])) ** 2
    tpsRK.append(tp)
    tp = np.abs(np.dot(y_trans, yM4[:,-1]))**2
    tpsM4.append(tp)
    
plt.figure(figsize=(6,6))
plt.title(r'$\xi$ ='+str(xi))
plt.plot(gamma_range, pex(gamma_range), '-k', label='exact')
plt.plot(gamma_range, tpsE, '-', color='fuchsia', marker = 's', label='Euler')
plt.plot(gamma_range, tpsRK, 'r->', label='RK4')
plt.plot(gamma_range, tpsM2, 'r--', label='Magnus-2')
plt.plot(gamma_range, tpsM4, 'b--', label='Magnus-4')
plt.xlabel('$\gamma$')
plt.ylabel('Transition Probability')
plt.legend()
plt.axis([0, 2 * np.pi, 0, 1])
plt.show()
