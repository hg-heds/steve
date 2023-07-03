import numpy as np
import statistics as stats

CONST = [-2,-1,1,2]

def rastrigin(X): # minimum 0 at (0,0,...,0)
    offset = 0
    shape = X.shape
    if len(shape)>1:
        return 10*shape[1] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)),axis=1)
    else: 
        return 10*shape[0] + np.sum((X+offset)**2-10*np.cos(2*np.pi*(X+offset)))

def DNLPSO(f=rastrigin, swarm_size=30, dim=2, max_epoch=400):

    xmin, xmax      = [-5.12,5.12]
    v_max           = (xmax-xmin) / 20
    swarm           = xmin + np.random.rand(swarm_size,dim) * (xmax-xmin)
    f_swarm         = f(swarm)
    v               = v_max * (2*np.random.rand(swarm_size,dim)-1)
    inertia_max     = 0.9
    inertia_min     = 0.4

    attratction_pbest = 2
    attratction_gbest = 2

    pbest           = swarm.copy()
    f_pbest         = f_swarm.copy()
    gbest           = swarm[np.argmin(f_swarm)]
    f_gbest         = f_swarm[np.argmin(f_swarm)]


    pc = 0.5 - 0.5 * ((np.arange(max_epoch))/(max_epoch-1))

    for epoch in range(max_epoch):


        inertia = inertia_max - epoch/max_epoch * (inertia_max-inertia_min)
 
        for i in range(swarm_size):
            for d in range(dim):
                pbest_f = pbest[i,d] if np.random.rand() > pc[epoch] else pbest[(i+CONST[np.random.randint(4)])%swarm_size,d]


                v[i,d] = inertia * v[i,d] + attratction_pbest * np.random.rand() * (pbest_f-swarm[i,d]) + attratction_gbest * np.random.rand() * (gbest[d]-swarm[i,d])
                v[i,d] = max(min(v[i,d],v_max),-v_max)
                swarm[i,d] = swarm[i,d]+v[i,d]
                swarm[i,d] = max(min(swarm[i,d],xmax),xmin)
            
        f_swarm = f(swarm)
        pbest[f_swarm<f_pbest] = swarm[f_swarm < f_pbest]
        f_pbest = f(pbest)
        argbest                 = np.argmin(f_pbest)
        gbest                   = pbest[argbest]
        f_gbest                 = f_pbest[argbest]

    return (gbest, f_gbest)


if __name__ == "__main__":
    print('DNLPSO')

    for dim in [2,5,10,30,50]:
        lst = [DNLPSO(rastrigin,dim=dim,swarm_size=50,max_epoch=100)[1] for _ in range(30)]
        print(f'{dim}\nMean: {stats.mean(lst)}\nStdev: {stats.stdev(lst)}\nMin: {min(lst)}\nMax: {max(lst)}\n')
