import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 10
print("Question 10 solutions")
h = 0.001
w = [-2]
ti = []
t = 1
ep = 0.0001

def dydt(t,y):
    return (y*y + y)/t

while t<3:
    h = min(h,3-t)
    k1 = h*dydt(t,w[-1])
    k2 = h*dydt(t+h/4, w[-1]+k1/4)
    k3 = h*dydt(t+3*h/8, w[-1]+3*k1/32+9*k2/32)
    k4 = h*dydt(t+12*h/13, w[-1]+1932*k1/2197-7200*k2/2197+7296*k3/2197)
    k5 = h*dydt(t+h, w[-1]+439*k1/216-8*k2+3680*k3/513-845*k4/4104)
    k6 = h*dydt(t+h/2, w[-1]-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)
    w1 = w[-1] + 25*k1/216+1408*k3/2565+2197*k4/4104-k5/5
    w2 = w[-1] + 16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55
    R = abs(w1-w2)/h
    if R==0:
        ti.append(t)
        t = t+h
        w.append(w1)
        h = h*1.2
    elif R<ep:
        ti.append(t)
        t = t+h
        w.append(w1)
        h = h*0.84*(ep/R)**(1/4)
    else: h = h*delta
ti.append(t)
t = np.linspace(1,3,20)
def y(t):
    return 2*t/(1-2*t)
plt.plot(ti,w, label = 'RK4 adaptive step')
plt.plot(t,y(t), label = 'analytical solution')
plt.vlines(ti, -2.3, -1.1, label = 'Mesh points', linestyles='dashed')
plt.title("Question 10 RK4 with adaptive step size")
plt.legend()
plt.show()
print('***************************************************\n')
