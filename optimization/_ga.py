# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 19:08:39 2021

@author: tang
"""

"""
========================================================================
:: parameters
func:           优化的目标函数
dims:           函数的维度
xlim:           变量取值范围(是一个元组或列表或可迭代对象, 有两个元素, 分别是下界和上界列表或元组)
                如[[-pi, -3, -9], [pi, 3, 9]], 或((-pi, -3, -9), (pi, 3, 9))
                默认为[[-1023]*dims, [1023]*dims]
population:     种群数目
variation:      变异率(不宜设置太大, 不然会变成随机搜索, 建议不要修改这个值, 默认5%(0.05))
percentage:     每次进化保留的种群个数, 默认保留50%(0.5)
max_iter:       最大迭代次数, 默认为100
slow_learn:     缓慢学习的次数, 防止长时间局部收敛, 默认允许一直缓慢学习, 默认为True
                设置为True时允许一直缓慢学习(即缓慢学习次数等于最大迭代次数),
                设置为一个正整数来表示最大缓慢学习次数, 如无必要建议一直缓慢学习
tol:            容忍度, 设置了缓慢学习次数才会生效, 此时建议设置slow_learn=10, 默认为10
                即超过10次没有迭代, 那么提前结束迭代, 默认tol=1e-4
float_length:   浮点数部分长度, 默认为64, 即1-64, 多数情况不需要修改
                ::<特别注意, 当设置为0时可以进行整数规划>
int_length:     整数部分长度, 会根据你的xlim最大整数部分进行自适应确定, 也可以手动设置
                若没有给出xlim, 则默认为10(即bin(1023)的长度), 建议设置xlim以获取恰当的长度
                (注:更长的长度能够获得更高的精度, 但是会损失编码速度)
verbose:        显示迭代过程, 默认不显示, 设置为 1 或 True 来显示
random_state:   随机种子, 默认为None
========================================================================
:: methods
fit():          开始执行GA算法
    pars: curve:绘制迭代曲线
          step: 显示迭代的步长, 设置的了verbose为真才会生效
========================================================================
:: attribute
best_:          最优解
minf_:          最优值
iter_:          迭代次数
========================================================================
:: example
def func(x):
    return np.sin(x[:, 1]) + np.cos(x[:, 0]) + x[:, 1]

dims = 2

ga = GeneticAlgorithm(func,
                      dims,
                      float_length=32,
                      xlim=((-5), (5)),
                      population=10,
                      max_iter=100,
                      verbose=1,
                      slow_learn=20,
                      percentage=0.5,
                      variation=0.01,
                      random_state=1)

ga.fit(step=1, curve=True)

第94次迭代, 当前最优个体适应度值:-9.873817338273549
第96次迭代, 当前最优个体适应度值:-9.872126459661212
第98次迭代, 当前最优个体适应度值:-9.846051270281427
========================================================================
"""

import numpy as np
from matplotlib import pyplot as plt

from .utils import roulette
from .utils import mat2bin
from .utils import bin2mat
from .utils import cross
from .utils import variation


class GeneticAlgorithm:
    def __init__(self, 
                 func,
                 dims,
                 *,
                 xlim=None,
                 population=50,
                 variation=0.05,
                 percentage=0.5,
                 max_iter=100,
                 tol=1e-4,
                 float_length=64,
                 int_length=None,
                 slow_learn=True,
                 verbose = 0,
                 random_state=None
                 ):
        self.func = func
        self.dims = dims
        self.xlim = xlim
        self.population = population
        self.variation = variation
        self.percentage = percentage
        self.max_iter = max_iter
        self.tol = tol
        self.int_length = int_length
        self.float_length = float_length
        self.slow_learn=slow_learn
        self.verbose = verbose
        self.random_state = random_state
        self.x = None
        # init
        np.random.seed(self.random_state)
        if self.xlim is None:
            self.int_length = 10
            self.xlim = [[-1023]*self.dims, [1023]*self.dims]
            self.x = np.random.uniform(self.xlim[0], self.xlim[1], (self.population, dims))
        else:            
            self.int_length = len(bin(int(np.max(np.abs(xlim))))) - 2
            # init x
            self.x = np.random.uniform(xlim[0], xlim[1], (self.population, dims))
        # attribute
        self.best_ = None
        self.minf_ = np.inf
        self.iter_ = None
        
    def fit(self, step=1, curve=False):
        hisf, slow_time = list(), 0
        for ite in range(self.max_iter):
            fval = self.func(self.x)
            # 选择
            idx = roulette(1, fval, 0.5)
            evx = self.x[idx, :]
            evx2bin = mat2bin(evx, self.int_length, self.float_length)
            # 交叉
            newx = cross(evx2bin, self.population)
            # 变异
            varx = variation(newx, self.percentage)
            # 更新
            self.x = bin2mat(varx)
            fmin = np.min(fval)
            idxf = np.argmin(fval)
            if fmin < self.minf_:
                self.minf_ = fmin
                self.best_ = self.x[idxf, :]
            hisf.append(fmin)
            self.iter_ = ite
            if self.verbose:
                if np.mod(ite, step) == 0:
                    print('第{}次迭代, 当前最优个体适应度值:{}'.format(ite, fmin))
            if self.slow_learn:
                continue
            else:
                if ite > 1 and slow_time < 10:
                    if np.abs(hisf[-1] - hisf[-2]) < self.tol:
                        slow_time += 1
                if slow_time >= self.slow_learn:
                    break
        if curve is True:
            self.plot_curve(hisf)
        return self.minf_, self.best_
    
    def plot_curve(self, hisf):
        plt.plot(hisf)
        plt.xlabel('iterations')
        plt.ylabel('function value')
        plt.show()
        return None








