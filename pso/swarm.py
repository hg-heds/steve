import numpy as np
from matplotlib import pyplot as plt
import statistics as stats
from functions import *
from itertools import product
np.set_printoptions(suppress=True,precision=2)
LOOP = True

def onclose(event):
    global LOOP 
    LOOP = False

def pso(f=rastrigin, swarm_size=100, dim=2, max_epoch=400, plot=False, coef=None):

    xmin, xmax      = f_range(f)
    v_max           = (xmax-xmin) / 20
    swarm           = xmin + np.random.rand(swarm_size,dim) * (xmax-xmin)
    f_swarm         = f(swarm)
    v               = v_max * (2*np.random.rand(swarm_size,dim)-1)

    inertia, attraction_pbest, attraction_gbest = coef if coef else 1, 0.1, 0.1

    pbest           = swarm.copy()
    f_pbest         = f_swarm.copy()
    gbest           = swarm[np.argmin(f_swarm)]
    f_gbest         = f_swarm[np.argmin(f_swarm)]


    if plot:

        fig         = plt.figure()
        ax          = fig.add_subplot(1,2,1)
        conv        = fig.add_subplot(1,2,2)
        major_ticks = np.arange(-5, 6, 1)
        convergence = [f_gbest]
        fig.canvas.mpl_connect('close_event', onclose)

    for epoch in range(max_epoch):

        if not LOOP: break

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

        # if epoch<4*max_epoch/5:
        #     attraction_pbest *= 

        if plot:

            convergence.append(f_gbest)
            ax.clear()
            ax.scatter(swarm[:,0],swarm[:,1])
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
        ax.scatter(swarm[:,0],swarm[:,1])
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
    
    results = []
    # for fun, func in enumerate([rastrigin, paraboloid, rosenbrock, griewank, ackley, styblinski_tang, schwefel, dixon_price, zakharov, levy]):
    for i,coef in enumerate(list(product(np.round(np.arange(0,1,0.1)+0.1,2),repeat=3))):
        if i%10 == 0: print(i,end=', ')
        for dim in [2,5,10]:
            lst = [pso(rastrigin,dim=dim,swarm_size=200,max_epoch=500)[1] for _ in range(30)]
            results.append([*coef,dim,stats.mean(lst),stats.stdev(lst),min(lst),max(lst)])
    np.save('results_rastrigin_200_500.npy',results)