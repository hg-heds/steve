from freelunch import benchmarks, PAO,PSO,QPSO
import numpy as np
import json
import statistics

metodos = {
            "QPSO":QPSO,
            "PSO":PSO,
            "PAO":PAO,
           }
otimos = {
            "QPSO":[],
            "PSO":[],
            "PAO":[],
          }
rastrigin = benchmarks.Ragstrin(n = 5)
for _ in range(30):
    for key,metodo in metodos.items():
        print(key)
        opt = metodo(obj=rastrigin.obj,bounds=rastrigin.bounds)
        result = opt(full_output=True)
        with open(f"{key}.json","w") as f:
            f.write(json.dumps(result,indent=2))

        otimo = opt()
        print(otimo)
        otimos[key].append(otimo)

print("\n\nOTIMO PAO",rastrigin.obj(np.array(statistics.mean(otimos["PAO"]))))
print("\nOTIMO PSO",rastrigin.obj(np.array(statistics.mean(otimos["PSO"]))))
print("\nOTIMO QPSO",rastrigin.obj(np.array(statistics.mean(otimos["QPSO"]))))

pass