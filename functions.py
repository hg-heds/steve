import numpy as np

def rastrigin(X): # minimum 0 at (0,0,...,0)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: 
        return 10*shape[0] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))

def paraboloid(X): # minimum 0 at (0,0,...,0)
    offset = 0
    if len(X.shape)>1: 
        return np.sum((X+offset)**2,axis=1)
    else: 
        return np.sum((X+offset)**2)

def rosenbrock(X): # minimum 0 at (1,1,...,1)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return np.sum(100*(X[:,1:]-X[:,:-1])**2+(X[:,:-1]+offset-1)**2,axis=1)
    else: 
        return np.sum(100*(X[1:]-X[:-1])**2+(X[:-1]+offset-1)**2)

def griewank(X): # minimum 0 at (0,0,...,0)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return np.sum((X+offset)**2/4000,axis=1) - np.prod(np.cos((X+offset)/np.sqrt(np.arange(shape[1])+1)),axis=1) + 1
    else: 
        return np.sum((X+offset)**2/4000) - np.prod(np.cos((X+offset)/np.sqrt(np.arange(shape[0])+1))) + 1

def ackley(X): # minimum 0 at (0,0,...,0)
    a,b,c = (20,0.2,2*3.1415926535)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return -a*np.exp(-b*np.sqrt(1/shape[1]*np.sum((X+offset)**2,axis=1))) -np.exp(1/shape[1]*np.sum(np.cos(c*(X+offset)),axis=1)) + a + np.exp(1)
    else: 
        return -a*np.exp(-b*np.sqrt(1/shape[0]*np.sum((X+offset)**2))) -np.exp(1/shape[0]*np.sum(np.cos(c*(X+offset)))) + a + np.exp(1)

def styblinski_tang(X):  # minimum -39.16599d at (-2.903534,...,-2.903534)
    offset = 0 
    shape = X.shape 
    if len(shape)>1:
        return 0.5 * np.sum((X+offset)**4-16*(X+offset)**2+5*(X+offset),axis=1)
    else: 
        return 0.5 * np.sum((X+offset)**4-16*(X+offset)**2+5*(X+offset))

def schwefel(X): # minimum 0 at (420.9687,420.9687,...,420.9687)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return 418.9829 * shape[1] - np.sum((X+offset)*np.sin(np.sqrt(np.abs(X+offset))),axis=1)
    else: 
        return 418.9829 * shape[0] - np.sum((X+offset)*np.sin(np.sqrt(np.abs(X+offset))))

def dixon_price(X): # minimum 0 at xi = 2**(-(2**i-2)/2**i)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return (X[:,0]-1)**2 + np.sum(np.arange(1,shape[1])*(2*(X[:,1:]+offset)**2-X[:,:-1]-offset)**2,axis=1)
    else: 
        return (X[:,0]-1)**2 + np.sum(np.arange(1,shape[0])*(2*(X[1:]+offset)**2-X[:-1]-offset)**2)

def zakharov(X): # minimum 0 at (0,0,...,0)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return np.sum((X+offset)**2,axis=1) + np.sum(0.5*np.arange(shape[1])*(X+offset),axis=1)**2 + np.sum(0.5*np.arange(shape[1])*(X+offset),axis=1)**4
    else: 
        return np.sum((X+offset)**2) + np.sum(0.5*np.arange(shape[0])*(X+offset))**2 + np.sum(0.5*np.arange(shape[0])*(X+offset))**4

def levy(X): # minimum 0 at (1,1,...,1)
    offset = 0
    shape = X.shape
    w = 1 + (X+offset-1)/4
    if len(shape)>1:
        return np.sin(np.pi*w[:,0])**2 + np.sum((X[:,:-1]-1)**2*(1+10*np.sin(np.pi*w[:,:-1]+1)**2)+(w[:,[-1]]-1)**2*(1+np.sin(2*np.pi*w[:,[-1]])),axis=1)
    else: 
        return np.sin(np.pi*w[0])**2 + np.sum((X[:-1]-1)**2*(1+10*np.sin(np.pi*w[:-1]+1)**2)+(w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])))

def f_range(f):

    if f ==        rastrigin:    return (  -5.12,   5.12)
    if f ==       paraboloid:    return (    -50,     50)
    if f ==       rosenbrock:    return (   -100,    100)
    if f ==         griewank:    return (   -600,    600)
    if f ==         schwefel:    return (   -500,    500)
    if f ==           ackley:    return (-32.768, 32.768)
    if f ==  styblinski_tang:    return (     -5,      5)
    if f ==      dixon_price:    return (    -10,     10)
    if f ==         zakharov:    return (     -5,     10)
    if f ==             levy:    return (    -10,     10)
    if                  True:    return (  -5.12,   5.12)

