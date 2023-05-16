import numpy as np

def rastrigin(X):
    offset = -1
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: return 10 + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))


def ga(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12]):
    xmin, xmax = lim
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    for _ in range(max_gen):
        new_pop = np.zeros([pop_size,dim])
        ##################################################################################### recombination
        for i in range(pop_size):
            r = np.random.randint(0,pop_size)
            new_pop[i,:] = (population[i,:] + population[r,:])/2 ############# mean
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
    return min(rastrigin(population))

def de(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12]):
    xmin, xmax = lim
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    for _ in range(max_gen):
        for i in range(pop_size):
            C = np.random.rand()+0.5
            locked = np.random.randint(dim)
            r1,r2,r3 = np.random.permutation(pop_size)[:3]
            child = population[r1] + C*(population[r3]-population[r2])
            rand = np.random.rand(dim)<0.5
            rand[locked] = False
            child[rand] = population[i][rand]
            if rastrigin(child) <= rastrigin(population[i]):
                population[i] = child
    return min(rastrigin(population))

def test():
    from time import time
    tm = time()
    DIM = 10
    SIZE = 100
    GEN = int(1e4/SIZE)
    lst_ga = [ga(pop_size=SIZE, dim=DIM, max_gen=GEN) for _ in range(30)]
    lst_de = [de(pop_size=SIZE, dim=DIM, max_gen=GEN) for _ in range(30)]
    import statistics as stats 
    print("Genetic Algorithm:\n\tMean: {}\n\tStandard Deviation: {}\n\tMaximum: {}\n\tMinimum: {}\nDifferential Evolution:\n\tMean: {}\n\tStandard Deviation: {}\n\tMaximum: {}\n\tMinimum: {}".format(stats.mean(lst_ga),stats.stdev(lst_ga),max(lst_ga),min(lst_ga),stats.mean(lst_de),stats.stdev(lst_de),max(lst_de),min(lst_de)))
    print(time()-tm)
for _ in range(10):
    test()