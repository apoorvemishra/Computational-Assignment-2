import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 11
print("Question 11 solutions")
h = 0.0001
t = np.arange(0,1+h,h)
w = np.zeros(int(1/h)+1)
w[0] = 1

def dwdt(ti, wi):
    return 1/(ti**2 + (wi*(1-ti))**2)

for i in range(0,int(1/h)):
    k1 = dwdt(t[i], w[i])
    k2 = dwdt(t[i]+h/2, w[i]+h*k1/2)
    k3 = dwdt(t[i]+h/2, w[i]+h*k2/2)
    k4 = dwdt(t[i+1], w[i]+h*k3)
    w[i+1] = w[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

plt.plot(t,w, label = 'solution plot with changed domain')
plt.title("Question 11 RK4 with infinite time domain")
y = 3.5*10**6
ti = y/(1+y)
plt.vlines(ti,1,2.2, linestyle = 'dashed', label = 't = 3.5 x 10^6')
print(f'The value of solution at t = {y} is {w[-1]}')
plt.legend()
plt.show()
print('***************************************************\n')
