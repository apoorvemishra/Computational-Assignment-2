
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp


#Question 1
print("Question 1 part 1 solutions")
h = 0.01
x = np.arange(0,1,h)
w = np.zeros(int(1/h))
w[0] = np.exp(1)
for i in range(len(x)-1):
    w[i+1] = w[i]/(1+9*h)
plt.plot(x,w, label = 'Backward integration for y\' = -9y')
plt.legend()
plt.title("Question 1 part 1 solutions")
plt.figure()


print("Question 1 part 2 solutions")
h = 0.01
x = np.arange(0,1,h)
w = np.zeros(int(1/h))
w[0] = 1/3
for i in range(len(x)-1):
    f = lambda z: z - w[i] - h*(2*x[i+1] - 20*(z - x[i+1])**2)
    w[i+1] = opt.newton(f,0)
plt.plot(x,w, label = 'Backward Euler for y\' = −20(y − x)\u00b2 + 2x')


#just checking the difference between backward and forward eular
w1 = np.zeros(int(1/h))
w1[0] = 1/3
for i in range(len(x)-1):
    w1[i+1] = w1[i] + h*(2*x[i] - 20*(w1[i]-x[i])*(w1[i]-x[i]))

plt.plot(x,w1, label = 'Forward Euler')
plt.legend()
plt.title("Question 1 part 2 solutions")
plt.show()
print('***************************************************\n')

