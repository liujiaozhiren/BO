# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from skopt import dummy_minimize, Optimizer
from tqdm import tqdm

M = 5
N = 20
f = open("output.txt", 'w')

B = np.random.uniform(10, 30, size=(N, M))
A = np.random.uniform(5, 20, size=(M))
v = np.random.uniform(25, 100, size=(N))
d = np.random.uniform(50, 300, size=(M))
b_a = [[B[i, j] - A[j] for j in range(M)] for i in range(N)]
MaxB_A = np.max(b_a) * 10.0
maxr = min(d.max(), v.max())

# b_a = np.random.rand(N, M)
# d = np.random.rand(M)
# v = np.random.rand(N)
# MaxB_A = b_a.max() * 2
# maxr = min(d.max(), v.max())
max_target = 0.0
print(MaxB_A, maxr, "\nD:", d, "\nV:", v, "\nB-A", b_a)


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


def noisy_valued(array, noisy_level=0.5):
    ar = [p + noisy_level * np.random.randn() for p in array]
    for i, item in enumerate(ar):
        if item < 0:
            ar[i] = 0
        elif item > maxr:
            ar[i] = maxr
    return ar


def opt():
    global max_target
    # 创建优化器
    opt = Optimizer(dimensions=[(0.0, maxr)] * (N * M))
    for epoch in range(1000000):
        # 迭代优化过程
        with tqdm(range(10), f"epoch_{epoch}") as tq:
            for _ in tq:
                # 生成参数建议
                x = opt.ask()
                # print(restrain2(x), restrain1(x), x)
                # 计算目标函数的值
                _x = noisy_valued(x)
                y = objective(_x)
                # tq_str = f'output:{-y:.5f}, re1:{restrain1(x):.3f}, re2:{restrain2(x):.3f}, x:{x}'
                # print(tq_str)
                # tq.set_postfix_str(tq_str)
                # 提供参数的评估结果

                max_target = max(-y, max_target)
                ret = opt.tell(_x, y)

        # 获取找到的最佳参数
        best_params = ret.x
        tq_str = f'最大值:{-objective(best_params):.5f}, 约束1:{restrain1(best_params):.3f}, 2:{restrain2(best_params):.3f}, 参数:{best_params}'
        f.write(tq_str + "\n")
        # print(tq_str)
        f.flush()
        if not restrain1(best_params) > 0 and not restrain2(best_params) > 0 and -objective(best_params) >= max_target:
            break
    f.write(f"================================\n")
    f.write(f"最佳参数:{best_params}\n")
    f.write(f"最大值:{-objective(best_params)}\n")
    f.write(f"约束{restrain2(best_params)}|{restrain1(best_params)}\n")
    print(f"================================\n")
    print(f"最佳参数:{best_params}\n")
    print(f"最大值:{-objective(best_params)}\n")
    print(f"约束{restrain2(best_params)}|{restrain1(best_params)}\n")
    print(MaxB_A, maxr, "\nD:", d, "\nV:", v, "\nB-A", b_a)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opt()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
