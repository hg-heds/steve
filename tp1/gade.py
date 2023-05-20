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

def de(pop_size = 100, dim = 2, max_gen = 400, lim = [-5.12,5.12], de_type = 'rand-1'):
    xmin, xmax = lim
    population = lim[0] + np.random.rand(pop_size,dim) * (xmax-xmin)
    f_pop = rastrigin(population)
    for _ in range(max_gen):
        for i in range(pop_size):
            eta = np.random.rand()+0.5
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
            ################################################################# mean-1
            elif de_type == 'mean-1':
                r2,r3 = np.random.permutation(pop_size)[:2]
                child = np.mean(population,axis=0) + eta*(population[r2]-population[r3])
            ################################################################# mean-to-best-1
            elif de_type == 'mean-to-best-1':
                best = np.argmin(f_pop)
                lambda_ = np.random.rand()
                r2,r3 = np.random.permutation(pop_size)[:2]
                child = population[i] + lambda_*(population[best]-population[i]) + eta*(population[r2]-population[r3])
            ################################################################# rand-2
            elif de_type == 'rand-2':
                eta2 = np.random.rand()+0.5
                r1,r2,r3,r4,r5 = np.random.permutation(pop_size)[:5]
                child = population[r1] + eta*(population[r2]-population[r4]) + eta2*(population[r3]-population[r5])

            rand = np.random.rand(dim)<0.5
            rand[locked] = False
            child[rand] = population[i][rand]
            if rastrigin(child) <= rastrigin(population[i]):
                population[i] = child
                f_pop[i] = rastrigin(child)
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