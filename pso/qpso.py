for d in [2,5,10,30,50]:
    qpso = PSO(benchmarks.Ragstrin(n=d),bounds=np.array([[-5.12,5.12]]*d))
    qpso.hypers.update({'N':50, 'G':500})
    runs = []
    for _ in range(30):
        runs.append(rastrigin(qpso())[0])
    print(f'''QPSO
Dim: {d}
Mean: {statistics.mean(runs)}
Stdev: {statistics.stdev(runs)}
Min: {min(runs)}
Max: {max(runs)}
''')
    data['pso_fr'][d] = {'dim':d, 'mean':statistics.mean(runs), 'stdev':statistics.stdev(runs),'min':min(runs),'max':max(runs)}
