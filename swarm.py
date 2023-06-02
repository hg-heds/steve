import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True,precision=2)
LOOP = True

def onclose(event):
    global LOOP 
    LOOP = False

def rastrigin(X):
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: return 10*shape[0] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))

def paraboloid(X):
    offset = 0
    if len(X.shape)>1: return np.sum((X+offset)**2,axis=1)
    else: return np.sum((X+offset)**2)

def rosenbrock(X):
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return np.sum(100*(X[:,1:]-X[:,:-1])**2+(X[:,:-1]+offset-1)**2,axis=1)
    else: return np.sum(100*(X[1:]-X[:-1])**2+(X[:-1]+offset-1)**2)

def griewank(X):
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return np.sum((X+offset)**2/4000,axis=1) - np.prod(np.cos((X+offset)/np.sqrt(np.arange(shape[1])+1)),axis=1) + 1
    else: return np.sum((X+offset)**2/4000) - np.prod(np.cos((X+offset)/np.sqrt(np.arange(shape[0])+1))) + 1

# def schwefel(X):
#     offset = 0
#     shape = X.shape
#     if len(shape)>1:
#         return 418.9829 * shape[1] - np.sum((X+offset)*np.sin(np.sqrt(np.abs(X+offset))),axis=1)
#     else: return 418.9829 * shape[0] - np.sum((X+offset)*np.sin(np.sqrt(np.abs(X+offset))))


fig = plt.figure()
fig.canvas.mpl_connect('close_event', onclose)
ax = fig.add_subplot(1,2,1)
conv = fig.add_subplot(1,2,2)
major_ticks = np.arange(-5, 6, 1)

f = rastrigin

pop_size        = 100
dim             = 10
xmin, xmax      = -5.12, 5.12
max_epoch       = 400
v_max           = (xmax-xmin) / 20
v_dist_max      = (xmax-xmin) / 10
pop             = xmin + np.random.rand(pop_size,dim) * (xmax-xmin)
f_pop           = f(pop)
v               = v_max * (2*np.random.rand(pop_size,dim)-1)
dist            = 10

inertia, attraction_pbest, attraction_gbest = 1, 0.1, 0.1

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
    

    v = inertia * v + attraction_pbest * (pbest - pop) + attraction_gbest*(gbest - pop)

    v[dist:,:][v[dist:,:]>v_max]              = v_max
    v[dist:,:][v[dist:,:]<-v_max]             = -v_max

    
    v[:dist,:][v[:dist,:]>v_max]              = v_dist_max
    v[:dist,:][v[:dist,:]<-v_max]             = -v_dist_max


    if epoch > max_epoch/2:
        v_max /= 1.006
        v_dist_max /= 1.002
        attraction_gbest *= 1.01

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
