import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 3
print("Question 3 solutions")
#RK4

h = 0.01
x = np.arange(0,1+h,h)
w = np.zeros(int(1/h)+1)
wp = np.zeros(int(1/h)+1)
w[0] = 0
wp[0] = 0

def f(t,y,yp):
    return np.array([t*np.exp(t) - t + 2*yp - y , yp])

for i in range(0,int(1/h)):
    k1 = f(x[i], w[i], wp[i])
    k2 = f(x[i]+h/2, w[i]+h*k1[1]/2, wp[i]+h*k1[0]/2)
    k3 = f(x[i]+h/2, w[i]+h*k2[1]/2, wp[i]+h*k2[0]/2)
    k4 = f(x[i+1], w[i]+h*k3[1], wp[i]+h*k3[0])
    
    wp[i+1], w[i+1] = [wp[i],w[i]] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

plt.plot(x,w, label = 'RK4')
plt.title("Question 3")
plt.legend()
plt.show()

print('***************************************************\n')
