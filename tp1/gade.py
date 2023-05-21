import numpy as np
from pandas import DataFrame, options
from time import time
import statistics as stats
from scipy.stats import cauchy
DEV = False
import warnings
warnings.filterwarnings("error")


options.display.float_format = '{:,.8f}'.format

def rastrigin(X):
    offset = -1
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: return 10 + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))

def ga(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12], rec_type = 'wid-offset-var'):
    if DEV: return 9
    xmin, xmax = lim
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    for _ in range(max_gen):
        ##################################################################################### recombination
        r = np.random.randint(0,pop_size,pop_size)
        if rec_type == 'mean':
            new_pop = (population + population[r])/2############# mean
        elif rec_type == 'wid-offset-var': 
            new_pop = population + (2*np.random.rand(pop_size,dim) - 0.5)*(population[np.random.randint(0,pop_size,pop_size)] - population)
        ##################################################################################### mutation
        mutation_prob = np.random.rand(pop_size)<0.5
        new_pop[mutation_prob,:] = new_pop[mutation_prob,:] + 0.5 * (np.random.random((pop_size,dim))[mutation_prob,:] - 0.5) * (xmax-xmin)
        new_pop[new_pop>xmax] = xmax
        new_pop[new_pop<xmin] = xmin
        ##################################################################################### selection
        temp_pop = np.concatenate((population, new_pop), axis=0)
        f_pop = rastrigin(temp_pop)
        index = np.argsort(f_pop)[:len(population)]
        population = temp_pop[index]
        population[population > xmax] = xmax
        population[population < xmin] = xmin
    return min(rastrigin(population))

def de(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12], de_type = 'rand-1'):
    xmin, xmax = lim
    if DEV: return 9
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    f_pop = rastrigin(population)
    for _ in range(max_gen):
        for i in range(pop_size):
            eta, eta2 = np.random.rand(2)+0.5
            locked = np.random.randint(dim)
            ################################################################# rand-1
            if de_type == 'rand-1':
                r1,r2,r3 = np.random.permutation(pop_size)[:3]
                child = population[r1] + eta*(population[r2]-population[r3])
            ################################################################# best-1
            elif de_type == 'best-1':
                best = np.argmin(f_pop)
                r2,r3 = np.random.permutation(pop_size)[:2]
                child = population[best] + eta*(population[r2]-population[r3])
            ################################################################# best-2
            elif de_type == 'best-2':
                best = np.argmin(f_pop)
                r2,r3,r4,r5 = np.random.permutation(pop_size)[:4]
                child = population[best] + eta*(population[r2]-population[r4]) + eta2*(population[r3]-population[r5])
            ################################################################# mean-1
            elif de_type == 'mean-1':
                r2,r3 = np.random.permutation(pop_size)[:2]
                child = np.mean(population,axis=0) + eta*(population[r2]-population[r3])
            ################################################################# current-to-best-1
            elif de_type == 'current-to-best-1':
                best = np.argmin(f_pop)
                r1,r2,r3 = np.random.permutation(pop_size)[:3]
                child = population[i] + eta*(population[best]-population[r1]) + eta2*(population[r2]-population[r3])
            ################################################################# current-to-pbest-1
            elif de_type == 'current-to-pbest-1':
                best = np.argmin(f_pop)
                r2,r3 = np.random.permutation(pop_size)[:2]
                child = population[i] + eta*(population[best]-population[i]) + eta2*(population[r2]-population[r3])
            ################################################################# rand-to-best-1
            elif de_type == 'rand-to-best-1':
                best = np.argmin(f_pop)
                r1,r2,r3 = np.random.permutation(pop_size)[:3]
                child = population[r1] + eta*(population[best]-population[r1]) + eta2*(population[r2]-population[r3])
            ################################################################# rand-2
            elif de_type == 'rand-2':
                r1,r2,r3,r4,r5 = np.random.permutation(pop_size)[:5]
                child = population[r1] + eta*(population[r2]-population[r4]) + eta2*(population[r3]-population[r5])

            rand = np.random.rand(dim)<0.5
            rand[locked] = False
            child[rand] = population[i][rand]
            if rastrigin(child) <= rastrigin(population[i]):
                population[i] = child
        population[population > xmax] = xmax
        population[population < xmin] = xmin
        f_pop = rastrigin(population)
    return min(rastrigin(population))

def sade(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12]): # Self-adaptive Differential Evolution Algorithm
    xmin, xmax = lim
    if DEV: return 9
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    f_pop = rastrigin(population)
    prob_rand = 0.5 # prob_ctb = 1 - prob_rand
    integrated = {'rand-1':0, 'current-to-best-1':0}
    discarded = {'rand-1':0, 'current-to-best-1':0}
    for gen in range(1,max_gen+1):
        if gen%50 == 0:
            # print(gen, prob_rand, integrated, discarded)
            ns1 = integrated['rand-1']
            ns2 = integrated['current-to-best-1']
            nf1 = discarded['rand-1']
            nf2 = discarded['current-to-best-1']
            try:
                prob_rand = ns1*(ns2+nf2)/(ns2*(ns1+nf1)+ns1*(ns2+nf2))
            except ZeroDivisionError :
                pass
            integrated = {'rand-1':0, 'current-to-best-1':0}
            discarded = {'rand-1':0, 'current-to-best-1':0}

        for i in range(pop_size):
            eta = 0.7*np.random.rand()+0.5
            locked = np.random.randint(dim)
            de_type = 'rand-1' if np.random.rand()<prob_rand else 'current-to-best-1'
            ################################################################# rand-1
            if de_type == 'rand-1':
                r1,r2,r3 = np.random.permutation(pop_size)[:3]
                child = population[r1] + eta*(population[r2]-population[r3])
            ################################################################# best-1
            elif de_type == 'current-to-best-1':
                best = np.argmin(f_pop)
                r2,r3 = np.random.permutation(pop_size)[:2]
                child = population[best] + eta*(population[r2]-population[r3])

            ##################### Recombination
            rand = np.random.rand(dim)<np.random.normal(0.5,0.1)
            rand[locked] = False
            child[rand] = population[i][rand]
            if rastrigin(child) <= rastrigin(population[i]):
                population[i] = child
                integrated[de_type] += 1
            else: discarded[de_type] += 1

        population[population > xmax] = xmax
        population[population < xmin] = xmin
        f_pop = rastrigin(population)
    return min(rastrigin(population))

def shade(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12]): # Success History Parameter Adaptation for Differential Evolution
    xmin, xmax = lim
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    f_pop = rastrigin(population)
    H = 10
    k = 0
    mcr = np.ones(H)*0.5
    mf = np.ones(H)*0.5
    for _ in range(max_gen):
        S = []
        for i in range(pop_size):
            r = np.random.randint(H)
            locked = np.random.randint(dim)
            cr = 0
            f = 0
            while cr <= 0: cr = min(np.random.normal(mcr[r],0.1),1)
            while f <= 0: f = min(cauchy.rvs(loc=mf[r],scale=0.1),1)
            best = np.argmin(f_pop)
            r2,r3 = np.random.permutation(pop_size)[:2]
            child = population[i] + f*(population[best]-population[i]) + f*(population[r2]-population[r3])
            rand = np.random.rand(dim)<cr
            rand[locked] = False
            child[rand] = population[i][rand]
            f_parent = rastrigin(population[i])
            f_child = rastrigin(child)
            if f_child < f_parent:
                df = abs(f_child-f_parent)
                population[i] = child
                S.append((cr,f,df))
        population[population > xmax] = xmax
        population[population < xmin] = xmin
        f_pop = rastrigin(population)
        if S:
            try:
                S = np.array(S)
                scr = S[:,0]
                sf = S[:,1]
                df = S[:,2]
                mcr[k] = np.sum(df/np.sum(df)*scr)
                mf[k] = np.sum(df/np.sum(df)*sf**2)/np.sum(df/np.sum(df)*sf)
                k = k+1
                if k == H: k = 0
            except:
                print(S)
    return min(rastrigin(population))


def test(DIM = 10, SIZE = 20):
    tm = time()
    GEN = int(1e4/SIZE)
    lst_gade = [
        [ga(pop_size=SIZE, dim=DIM, max_gen=GEN) for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'rand-1') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'best-1') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'best-2') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'mean-1') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'rand-to-best-1') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'current-to-best-1') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'current-to-pbest-1') for _ in range(30)],
        [de(pop_size=SIZE, dim=DIM, max_gen=GEN, de_type = 'rand-2') for _ in range(30)],
        [sade(pop_size=SIZE, dim=DIM, max_gen=GEN) for _ in range(30)],
        [shade(pop_size=SIZE, dim=DIM, max_gen=GEN) for _ in range(30)]
    ]
    df = [
        [stats.mean(lst),stats.stdev(lst),max(lst),min(lst)] for lst in lst_gade
    ]
    df = DataFrame(df,['GA','DE_rand-1','DE_best-1','DE_best-2','DE_mean-1','DE_rand-to-best-1','DE_current-to-best-1','DE_current-to-pbest-1','DE_rand-2','sade','shade'],['mean','desvio padrão', 'max','min'])
    print('\n\nDim: '+str(DIM)+'\t Size: '+str(SIZE))
    print(df.to_string())
    print(time()-tm)
for size in [100,25,20,10]:
    for dim in [2,5,10,12]:
        test(DIM = dim, SIZE=size)  