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

def PSO_sono(f=rastrigin, swarm_size=50, dim=2, max_epoch=100, plot=False, coef=None):

    xmin, xmax      = f_range(f)
    v_max           = (xmax-xmin) / 20
    swarm           = xmin + np.random.rand(swarm_size,dim) * (xmax-xmin)
    f_swarm         = f(swarm)
    v               = v_max * (2*np.random.rand(swarm_size,dim)-1)

    omega_max       = 0.9
    omega_min       = 0.4

    pbest           = swarm.copy()
    f_pbest         = f_swarm.copy()
    gbest           = swarm[np.argmin(f_swarm)]
    f_gbest         = f_swarm[np.argmin(f_swarm)]

    r = 0
    G = 1


    if plot:

        fig         = plt.figure()
        ax          = fig.add_subplot(1,2,1)
        conv        = fig.add_subplot(1,2,2)
        major_ticks = np.arange(-5, 6, 1)
        convergence = [f_gbest]
        fig.canvas.mpl_connect('close_event', onclose)

    for epoch in range(max_epoch):

        if not LOOP: break

        sort = np.argsort(f_swarm)
        f_swarm = f_swarm[sort]
        swarm = swarm[sort]

        omega = omega_max - epoch/max_epoch * (omega_max-omega_min)
        c1 = 2.5 - 2*epoch/max_epoch
        c2 = 0.5 + 2*epoch/max_epoch
        


        swarm                   = swarm + v 
        swarm[swarm>xmax]       = xmax
        swarm[swarm<xmin]       = xmin
        f_swarm                 = f(swarm)
        pbest[f_swarm<f_pbest]  = swarm[f_swarm<f_pbest]
        f_pbest                 = f(pbest)
        argbest                 = np.argmin(f_pbest)
        gbest                   = pbest[argbest]
        f_gbest                 = f_pbest[argbest]
 
        v =  v + attraction_pbest * np.random.rand(swarm_size, 1) * (pbest - swarm) + attraction_gbest * np.random.rand(swarm_size, 1) * (gbest - swarm)

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
    print('PSO')
    for dim in [2,5,10]:
        lst = [PSO(rastrigin,dim=dim,swarm_size=50,max_epoch=100)[1] for _ in range(30)]
        print(f'{dim}\nMean:{stats.mean(lst)}\nStdev: {stats.stdev(lst)}\nMin: {min(lst)}\nMax:{max(lst)}\n\n')