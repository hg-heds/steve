import numpy as np
from matplotlib import pyplot as plt, animation as ani
pop_size = 100
pop_dimension = 1
max_generations = 100
limits = [-100,100]
population = limits[0] + np.random.rand(pop_size,pop_dimension) * (limits[1]-limits[0])
def paraboloid(X):
    return np.sum(X**2,axis=1)
def rastrigin(X):
    dim = X.shape[1]
    return 10*dim + np.sum(X**2-10*np.cos(2*np.pi*X),axis=1)

funcs = {
    "paraboloid":paraboloid,
    "rastrigin":rastrigin
}
func = "rastrigin"

F = funcs[func]

def rec_mean(a,b):
    return (a + b)/2

recombination_types = {
    "mean":rec_mean
}

def recombination(rec_type = 'mean'):
    rec = recombination_types[rec_type]
    new_population = np.zeros([pop_size,pop_dimension])
    for i in range(pop_size):
        r = np.random.randint(0,pop_size)
        new_population[i,:] = rec(population[i,:], population[r,:])
    return new_population

def elitism(pop_obj,size):
    index = np.argsort(pop_obj)[:size]
    return index

def tournment(pop_obj,size):
    pass

def selection(population,new_population,selection_type):
    temp_pop = np.concatenate((population, new_population), axis=0)
    objective = F(temp_pop)
    index = selection_type(objective,pop_size)
    population = temp_pop[index,:]
    return population





fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def animate(i):
    global population
    ax.clear()
    ax.set_ylim([0,100])
    ax.set_xlim([-5,5])
    ax.plot(population,F(population),'.')
    population = selection(population,recombination("mean"),elitism)

    
animation = ani.FuncAnimation(fig, animate, interval=100) 
plt.show()
