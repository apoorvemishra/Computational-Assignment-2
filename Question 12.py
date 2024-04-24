import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 12
print("Question 12 solutions")
h = 0.01
t = np.arange(0,1+h,h)
w = np.zeros((int(1/h)+1,3))
w[0] = [3,-1,1]

def dwdt(ti, wi):
    w1 = wi[0] + 2*wi[1] - 2*wi[2] + np.exp(-ti)
    w2 = wi[1] + wi[2] - 2*np.exp(-ti)
    w3 = wi[0] + 2*wi[1] + np.exp(-ti)
    return np.array([w1,w2,w3])

for i in range(0,int(1/h)):
    k1 = dwdt(t[i], w[i])
    k2 = dwdt(t[i]+h/2, w[i]+h*k1/2)
    k3 = dwdt(t[i]+h/2, w[i]+h*k2/2)
    k4 = dwdt(t[i+1], w[i]+h*k3)
    
    w[i+1] = w[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

plt.plot(t,w[:,0], label = 'u1')
plt.plot(t,w[:,1], label = 'u2')
plt.plot(t,w[:,2], label = 'u3')
plt.title("Question 12 RK4")
plt.legend()
plt.show()
print('***************************************************\n')
