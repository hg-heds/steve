import numpy as np
from matplotlib import pyplot as plt
import statistics as stats
np.set_printoptions(suppress=True,precision=2)
LOOP = True

def onclose(event):
    global LOOP 
    LOOP = False

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

def f_range(f):

    if f == rastrigin:          return (-5.12,5.12)
    if f == paraboloid:         return (-50,50)
    if f == rosenbrock:         return (-100,100)
    if f == griewank:           return (-600,600)
    if f == schwefel:           return (-500,500)
    if f == ackley:             return (-32.768,32.768)
    if f == styblinski_tang:    return (-5,5)
    if f == dixon_price:        return (-10,10)
    if f == zakharov:           return (-5,10)

def swarm(f=rastrigin, pop_size=100, dim=2, max_epoch=400, plot=False):

    if plot:
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', onclose)
        ax = fig.add_subplot(1,2,1)
        conv = fig.add_subplot(1,2,2)
        major_ticks = np.arange(-5, 6, 1)

    xmin, xmax      = f_range(f)
    v_max           = (xmax-xmin) / 20
    pop             = xmin + np.random.rand(pop_size,dim) * (xmax-xmin)
    f_pop           = f(pop)
    v               = v_max * (2*np.random.rand(pop_size,dim)-1)

    inertia, attraction_pbest, attraction_gbest = np.random.rand(), 0.1, 0.1

    pbest           = pop.copy()
    f_pbest         = f_pop.copy()
    gbest           = pop[np.argmin(f_pop)]
    f_gbest         = f_pop[np.argmin(f_pop)]
    convergence     = [f_gbest]

    for epoch in range(max_epoch):
        if not LOOP: break
        pop                     = pop + v 
        pop[pop>xmax]           = xmax
        pop[pop<xmin]           = xmin
        f_pop                   = f(pop)
        pbest[f_pop<f_pbest]    = pop[f_pop<f_pbest]
        f_pbest                 = f(pbest)
        argbest                 = np.argmin(f_pbest)
        gbest                   = pbest[argbest]
        f_gbest                 = f_pbest[argbest]
        convergence.append(f_gbest)
        
        v = inertia * v + attraction_pbest * np.random.rand(pop_size, 1) * (pbest - pop) + attraction_gbest * np.random.rand(pop_size, 1) * (gbest - pop)

        v[v>v_max]              = v_max
        v[v<-v_max]             = -v_max

        if plot:
            ax.clear()
            ax.scatter(pop[:,0],pop[:,1])
            ax.plot(gbest[0],gbest[1],'ro')
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(xmin,xmax)
            fig.suptitle('[' + ' '.join(['%2.3f'%var for var in gbest]) + ']    f: %.3f'%f_gbest)
            conv.set_ylim(None,2*sum(convergence)/len(convergence))
            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)

            conv.plot(convergence,'g-')
            # plt.waitforbuttonpress()
            plt.pause(0.00001)

    if plot:
        ax.clear()
        ax.scatter(pop[:,0],pop[:,1])
        ax.plot(gbest[0],gbest[1],'ro')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(xmin,xmax)
        ax.set_title(f"Best: {np.round(gbest,3)}, f: {np.round(f_gbest,4)}")
        ax.set_title('[' + ' '.join(['%.3f'%var for var in gbest]) + '] f: %.3f'%f_gbest)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        plt.show(block=True)
    return (gbest, f_gbest)


if __name__ == "__main__":
    for dim in [2,5,10]:
        lst = [swarm(zakharov,dim=dim,pop_size=500,max_epoch=1000)[1] for _ in range(30)]
        print(f'# de Variáveis: {dim}\nMédia: {stats.mean(lst)}\nDesvio Padrão: {stats.stdev(lst)}\nMin: {min(lst)}\n')