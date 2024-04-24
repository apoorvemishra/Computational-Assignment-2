import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 14
print("Question 14 solutions")
h = 0.001
t = np.arange(1,2+h,h)
w = np.zeros(int(1/h)+1)
wp = np.zeros(int(1/h)+1)
w[0] = 1
wp[0] = 0

for i in range(0,int(1/h)):
    wp[i+1] = wp[i] + h*(t[i]*np.log(t[i]) + 2*wp[i]/t[i] - 2*w[i]/t[i]/t[i])
    w[i+1] = w[i] + h*wp[i]



def y(t):
    return 7*t/4 + np.log(t)*(t**3)/2 - (3*t**3)/4
plt.plot(t,y(t), label = "Analytical")
plt.plot(t,w, label = "Euler's method")
plt.title("Question 14")
plt.legend()
plt.show()
print('***************************************************\n')


