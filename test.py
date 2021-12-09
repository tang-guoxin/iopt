# -*- coding: utf-8 -*-


from optimization import GeneticAlgorithm
from optimization import ParticleSwarmOptimization

import numpy as np

def func(x):
    return np.sin(x[:, 1]) + np.cos(x[:, 0]) + x[:, 1]

dims = 2

ga = GeneticAlgorithm(func,
                      dims,
                      float_length=64,
                      xlim=((-5, -5), (5, 5)),
                      population=10,
                      max_iter=100,
                      verbose=0,
                      slow_learn=20,
                      percentage=0.5,
                      variation=0.01,
                      random_state=1
                      )

ga.fit(display=False, curve=True)



pso = ParticleSwarmOptimization(func,
                                dims, 
                                xlim=[[-5, -5], [5, 5]],
                                vlim=[[-1, -2], [2, 2]]
                                )


pso.fit(display=False, curve=True)


print(ga.minf_)
print(ga.best_)

print(pso.minf_)
print(pso.best_)









