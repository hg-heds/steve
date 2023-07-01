import numpy as np
from matplotlib import pyplot as plt
import statistics as stats
from functions import *
from itertools import product
np.set_printoptions(suppress=True,precision=2)
STOP = False

def onclose(event):
    global STOP 
    STOP = True


def select(f_swarm,i):
    rand = np.random.permutation(len(f_swarm))[:3]
    rand = rand[rand != i]
    r1,r2 = rand[0],rand[1]
    if f_swarm[r1] > f_swarm[r2]: return r2 
    return r1



def CLPSO_simple(f=rastrigin, swarm_size=30, dim=2, max_epoch=400, plot=False, coef=None):

    xmin, xmax      = f_range(f)
    v_max           = (xmax-xmin) / 20
    swarm           = xmin + np.random.rand(swarm_size,dim) * (xmax-xmin)
    f_swarm         = f(swarm)
    v               = v_max * (2*np.random.rand(swarm_size,dim)-1)
    inertia_max     = 0.9
    inertia_min     = 0.4

    c               = 2

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


    pc = 0.05 + 0.45 * (np.exp(10*(np.arange(swarm_size)-1)/(swarm_size-1))-1) / (np.exp(10)-1)

    for epoch in range(max_epoch):
        if STOP: break

        swarm                   = swarm + v 
        swarm[swarm>xmax]       = xmax
        swarm[swarm<xmin]       = xmin
        f_swarm                 = f(swarm)
        pbest[f_swarm<f_pbest]  = swarm[f_swarm<f_pbest]
        f_pbest                 = f(pbest)
        argbest                 = np.argmin(f_pbest)
        gbest                   = pbest[argbest]
        f_gbest                 = f_pbest[argbest]
        inertia                 = inertia_max - epoch/max_epoch * (inertia_max-inertia_min)
 

        for i in range(len(v)):
            for d in range(len(v[i])):
                pbest_f = pbest[i,d] if np.random.rand() > pc[i] else pbest[select(f_swarm,i),d]
                v[i,d] = inertia * v[i,d] + c * np.random.rand() * (pbest_f-swarm[i,d])



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
    print('CLPSO')
    for dim in [2,5,10]:
        lst = [CLPSO_simple(rastrigin,dim=dim,swarm_size=50,max_epoch=100)[1] for _ in range(30)]
        print(f'{dim}\nMean: {stats.mean(lst)}\nStdev: {stats.stdev(lst)}\nMin: {min(lst)}\nMax: {max(lst)}\n')
