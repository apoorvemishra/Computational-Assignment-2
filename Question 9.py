import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 8
print("Question 8 solutions")

def fun1(t,y):
    return t*np.exp(3*t) - 2*y
def exact1(t):
    return np.exp(3*t)*(5*t-1)/25 + np.exp(-2*t)/25
t1 = [0,1]
t = np.linspace(0,1,20)
y1 = [0]
sol1 = ivp(fun1,t1,y1, t_eval = t)
plt.plot(t,exact1(t), label = 'Analytical solution')
plt.plot(sol1.t, sol1.y[0],label = 'Scipy integrate solution' )
plt.legend()
plt.title('Question 8 First function')
plt.figure()

def fun2(t,y):
    return 1 - (t-y)**2
def exact2(t):
    return (t*t-3*t+1)/(t-3)
t2 = [2,2.99]
t = np.arange(2,3,0.01)
y2 = [1]
sol2 = ivp(fun2,t2,y2, t_eval = t)
plt.plot(t,exact2(t), label = 'Analytical solution')
plt.plot(sol2.t, sol2.y[0],label = 'Scipy integrate solution')
plt.legend()
plt.title('Question 8 Second function')
plt.figure()

def fun3(t,y):
    return 1+y/t
def exact3(t):
    return t*(np.log(t) + 2)
t3 = [1,2]
t = np.linspace(1,2,20)
y3 = [2]
sol3 = ivp(fun3,t3,y3, t_eval = t)
plt.plot(t,exact3(t), label = 'Analytical solution')
plt.plot(sol3.t, sol3.y[0],label = 'Scipy integrate solution')
plt.legend()
plt.title('Question 8 Third function')
plt.figure()

def fun4(t,y):
    return np.cos(2*t) + np.sin(3*t)
def exact4(t):
    return (1/6)*(-2*np.cos(3*t) + 3*np.sin(2*t) + 8)
t4 = [0,1]
t = np.linspace(0,1,20)
y4 = [1]
sol4 = ivp(fun4,t4,y4, t_eval = t)
plt.plot(t,exact4(t), label = 'Analytical solution')
plt.plot(sol4.t, sol4.y[0],label = 'Scipy integrate solution')
plt.legend()
plt.title('Question 8 Fourth function')
plt.show()

print('For all four ODE\'s scipy integrate ivp worked but have to include discretization separately')

print('***************************************************\n')
