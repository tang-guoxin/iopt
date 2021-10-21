from matplotlib import pyplot as plt
import numpy as np

""":一个更加完善的粒子群算法
input:
    func    : function 句柄
    dims    : 优化目标函数的维度
    xlim    : 自变量取值范围，是一个元组或列表，有两个元素，每个元素也是一个列表或元组
    vlim    : 速度的取值范围， 输入同 xlim
    pap     : 种群大小，默认为 20
    w       : 动量系数，建议 w = (0.09, 0.99)
    c1      : 学习因子1，即往全局最优靠近的趋势
    c2      : 学习因子2，即保持自身飞行速度不变的趋势
    max_iter: 最大迭代次数，默认100
    slow    : 缓慢学习的次数，默认不设置，此时等于最大迭代次数
    tol     : 容忍度，设置了slow才会生效，允许提前结束的最小误差

methods:
    class.fit()     : 开始搜索最小值，并返回最小值
    class.best      : 最优解
    class.curve()   : 绘制迭代曲线
    
notice:
    必要时，请注意检查目标函数应该有两个axis，以保证输入为向量和矩阵都能运算
    
example:
    见 __main__ 函数
"""


class PSO:
    def __init__(self,
                 func,
                 dims,
                 *,
                 xlim=(),
                 vlim=(),
                 pap=20,
                 w=(0.09, 0.99),
                 c1=1.65,
                 c2=1.65,
                 max_iter=100,
                 slow=None,
                 tol=1e-6
                 ):
        self.func = func
        self.dims = dims
        self.pap = pap
        self.xlim = xlim
        self.vlim = vlim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.tol = tol
        self.best = None
        self.x = np.random.uniform(xlim[0], xlim[1], (pap, dims))
        self.v = np.random.uniform(vlim[0], vlim[1], (pap, dims))
        self.his = []
        if slow is None:
            slow = max_iter
        self.slow = slow

    def condition(self, x, limit):
        c = np.zeros_like(x)
        for d in range(self.dims):
            tmp = x[:, d]
            tmp[tmp < limit[0][d]] = limit[0][d]
            tmp[tmp > limit[1][d]] = limit[1][d]
            c[:, d] = tmp
        return c

    def refresh_v(self, leader, follow, iter) -> bool:
        self.v = self.v + self.c1 * np.random.rand() * (leader - self.x) + self.c2 * np.random.rand() * (
                    follow - self.x)
        dert = self.w[0] + (self.w[1] - self.w[0]) * (self.max_iter - iter) / self.max_iter
        self.v = dert * self.v
        self.v = self.condition(x=self.v, limit=self.vlim)
        return True

    def refresh_follow(self, y, follow):
        for i in range(self.pap):
            if self.func(self.x[i, :]) < y[i]:
                follow[i, :] = self.x[i, :]
        return follow

    def curve(self):
        plt.plot(self.his, 'r--.')
        plt.show()
        return True

    def fit(self, display=False):
        times = 0
        follow = self.x
        y = self.func(follow)
        idx = np.argmin(y)
        leader = follow[idx, :]
        self.his.append(y[idx])
        for i in range(self.max_iter):
            self.refresh_v(leader=leader, follow=follow, iter=i)
            self.x = self.v + self.x
            self.x = self.condition(x=self.x, limit=self.xlim)
            follow = self.refresh_follow(y, follow=follow)
            y = self.func(follow)
            idx = np.argmin(y)
            leader = follow[idx, :]
            self.his.append(y[idx])
            if display:
                print('iter = %d. min = %f' % (i + 1, y[idx]))
            if np.abs(self.his[-1] - self.his[-2]) <= self.tol:
                times += 1
            if np.abs(self.his[-1] - self.his[-2]) > self.tol:
                times = 0
            if times > self.slow:
                break
        self.best = leader
        return np.min(y)


def fx(x):
    axis = 0
    if len(x.shape) > 1:
        axis = 1
    else:
        axis = 0
    return np.sum((x ** 2) - 9, axis=axis) + np.sin(np.sum((x ** 2) - 9, axis=axis)) + np.cos(np.sum((x ** 2) - 9, axis=axis))


if __name__ == '__main__':
    ser = PSO(fx, 2, xlim=([-5, -5], [5, 5]), vlim=([-2, -2], [2, 2]))

    fmin = ser.fit(display=True)

    ser.curve()

    print(fmin)

    print(ser.best)
