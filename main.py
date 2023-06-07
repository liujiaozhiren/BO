# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from skopt import dummy_minimize, Optimizer
from tqdm import tqdm

M = 4
N = 3

b_a = np.random.rand(N, M)
d = np.random.rand(M)
v = np.random.rand(N)
MaxB_A = b_a.max() * 2
maxr = max(d.max(), v.max())

print(d, v, MaxB_A, maxr)


# 目标函数

def objective(input):
    x_r = np.array(input).reshape((N, M))
    sum = np.sum(x_r * b_a)
    sum -= restrain1(x_r) * MaxB_A
    sum -= restrain2(x_r) * MaxB_A
    return -sum


def restrain1(x_r):
    if type(x_r) != np.ndarray:
        x_r = np.array(x_r).reshape((N, M))
    tmp = x_r.sum(axis=0)
    tmp = tmp - d
    ret = np.maximum(tmp, 0).sum()
    return ret


def restrain2(x_r):
    if type(x_r) != np.ndarray:
        x_r = np.array(x_r).reshape((N, M))
    tmp = x_r.sum(axis=1)
    tmp = tmp - v
    ret = np.maximum(tmp, 0).sum()
    return ret


def opt():
    # 创建优化器
    opt = Optimizer(dimensions=[(0.0, maxr)] * (N * M))
    while True:
        # 迭代优化过程
        with tqdm(range(50)) as tq:
            for _ in tq:
                # 生成参数建议
                x = opt.ask()
                # print(restrain2(x), restrain1(x), x)
                # 计算目标函数的值
                y = objective(x)
                #tq_str = f'output:{-y:.5f}, re1:{restrain1(x):.3f}, re2:{restrain2(x):.3f}, x:{x}'
                #print(tq_str)
                # tq.set_postfix_str(tq_str)
                # 提供参数的评估结果
                opt.tell(x, y)

        # 获取找到的最佳参数
        best_params = opt.ask()
        tq_str = f'最大值:{-objective(best_params):.5f}, 约束1:{restrain1(best_params):.3f}, 2:{restrain2(best_params):.3f}, 参数:{best_params}'
        print(tq_str)
        if not restrain1(best_params) > 0 and not restrain2(best_params)> 0:
            break
    print("最佳参数:", best_params)
    print("最大值:", -objective(best_params))
    print("约束", restrain2(best_params), restrain1(best_params))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opt()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
