import numpy as np
import statistics as stats
from scipy.stats import cauchy

def rastrigin(X):
    offset = -1
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: return 10*shape[0] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))

def shade(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12]): # Success History Parameter Adaptation for Differential Evolution
    xmin, xmax = lim
    population = xmin + np.random.rand(pop_size,dim) * (xmax-xmin)
    f_pop = rastrigin(population)
    H = 10
    k = 0
    mcr = np.ones(H)*0.5
    mf = np.ones(H)*0.5
    for _ in range(max_gen):
        r = np.random.randint(0,H,pop_size)
        locked = np.random.randint(0,dim,pop_size)
        cr = np.zeros(pop_size)
        f = np.zeros(pop_size)
        while (cr<=0).any(): 
            cr[cr<=0] = np.array([np.random.normal(loc=mcr[ri],scale=0.1) for ri in r[cr<=0]])
        cr[cr>1] = 1
        while (f<=0).any():
            f[f<=0] = np.array([cauchy.rvs(loc=mf[ri],scale=0.1) for ri in r[f<=0]])
        f[f>1] = 1
        best = np.argmin(f_pop)
        r1 = np.random.randint(0,pop_size,pop_size)
        r2 = np.random.randint(0,pop_size,pop_size)
        while (r1==r2).any():
            r2[r1==r2] = np.random.randint(0,pop_size,np.sum(r1==r2))

        children = population + np.expand_dims(f,axis=1)*(population[best]-population) + np.expand_dims(f,axis=1)*(population[r1]-population[r2])
        rand = np.random.rand(pop_size,dim)<np.expand_dims(cr,axis=1)
        rand[np.arange(pop_size),locked] = False
        children[rand] = population[rand]
        f_parents = rastrigin(population)
        f_children = rastrigin(children)
        df = abs(f_children-f_parents)
        integrate = f_children < f_parents
        population[integrate] = children[integrate] 
        population[population > xmax] = xmax
        population[population < xmin] = xmin
        if integrate.any():
            scr=cr[integrate]
            sf=f[integrate]
            sdf=df[integrate]
            mcr[k] = np.sum(sdf/np.sum(sdf)*scr)
            mf[k] = np.sum(sdf/np.sum(sdf)*sf**2)/np.sum(sdf/np.sum(sdf)*sf)
            k = (k+1)%H
        f_pop = rastrigin(population)
    return min(rastrigin(population))


pop_size = 20
gen = int(1e4/pop_size)

for dim in [2,5,10]:
    lst = [shade(pop_size=pop_size, dim=dim, max_gen=gen) for _ in range(30)]
    print(f'# de Variáveis: {dim}\nMédia: {stats.mean(lst)}\nDesvio Padrão: {stats.stdev(lst)}\n')
     