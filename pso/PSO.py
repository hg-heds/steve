import numpy as np
import statistics as stats

def rastrigin(X): # minimum 0 at (0,0,...,0)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: 
        return 10*shape[0] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))



def PSO(f=rastrigin, swarm_size=100, dim=2, max_epoch=400):

    xmin, xmax      = [-5.12,5.12]
    v_max           = (xmax-xmin) / 20
    swarm           = xmin + np.random.rand(swarm_size,dim) * (xmax-xmin)
    f_swarm         = f(swarm)
    v               = v_max * (2*np.random.rand(swarm_size,dim)-1)

    attraction_pbest, attraction_gbest = 2,2
    inertia_max     = 0.9
    inertia_min     = 0.4
    inertia         = inertia_max
    pbest           = swarm.copy()
    f_pbest         = f_swarm.copy()
    gbest           = swarm[np.argmin(f_swarm)]
    f_gbest         = f_swarm[np.argmin(f_swarm)]

    for epoch in range(max_epoch):

        inertia = inertia_max - epoch/max_epoch * (inertia_max-inertia_min)
        swarm                   = swarm + v 
        swarm[swarm>xmax]       = xmax
        swarm[swarm<xmin]       = xmin
        f_swarm                 = f(swarm)
        pbest[f_swarm<f_pbest]  = swarm[f_swarm<f_pbest]
        f_pbest                 = f(pbest)
        argbest                 = np.argmin(f_pbest)
        gbest                   = pbest[argbest]
        f_gbest                 = f_pbest[argbest]
 
        v = inertia * v + attraction_pbest * np.random.rand(swarm_size, 1) * (pbest - swarm) + attraction_gbest * np.random.rand(swarm_size, 1) * (gbest - swarm)

        v[v>v_max]              = v_max
        v[v<-v_max]             = -v_max


    return (gbest, f_gbest)


if __name__ == "__main__":
    for dim in [2,5,10,30,50]:
        lst = [PSO(rastrigin,dim=dim,swarm_size=50,max_epoch=100)[1] for _ in range(30)]
        print(f'{dim}\nMean:{stats.mean(lst)}\nStdev: {stats.stdev(lst)}\nMin: {min(lst)}\nMax:{max(lst)}\n')

