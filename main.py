from random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as rmse
import numpy as np
from numpy.random import choice, randint
from numpy.linalg import matrix_rank, svd, norm, inv
from scipy.linalg import diagsvd
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn import linear_model

SEED = 179

np.random.seed(SEED)
seed(SEED)

###########################
### A L G O R I T H M S ###
###########################

def GD(A, b, x_0, r, eta, iter_num):
    x_i = [x_0]
    A_t = np.transpose(A)

    for i in range(iter_num):
        nabla = np.matmul(A_t, np.matmul(A, x_i[-1])-b)
        new_point = x_i[-1] - eta * nabla
        if(norm(new_point)> r):
            x_i.append(new_point/norm(new_point) * r)
        else:
            x_i.append(new_point)
        if nabla <= 0.00001:
            break
    return x_i[-1]

def SGD_SO_f(A,b,w):
    idx = choice(range(A.shape[0]))
    A_part = A[idx]
    A_t_part = np.transpose(A_part)
    b_part = b[idx]

    return np.dot(A_t_part, np.array(np.matmul(A_part,w) - b_part))

def SGD(A, b, x_0, r, alpha, iter_num):

    x_i = [x_0]
    y_s = [x_i[-1]]

    for k in range(iter_num):
        eta = 2 / (alpha * (k+1))
        ch_num = A.shape[0]
        for s in range(ch_num):
            new_point = y_s[-1] - eta * SGD_SO_f(A,b,y_s[-1])
            if (norm(new_point) > r):
                y_s.append(new_point / norm(new_point) * r)
            else:
                y_s.append(new_point)
        x_new = 0
        for k, y_k in enumerate(y_s):
            x_new += 2 * (k+1) / (ch_num * (ch_num+1)) * y_k
        x_i.append(x_new)
        y_s = [y_s[-1]]
    return x_i

def SGD_MB_SO_f(A,b,w, b_size):
    idx = choice(range(A.shape[0]), b_size)
    A_part = A[idx]
    A_t_part = np.transpose(A_part)
    b_part = b[idx]

    return np.dot(A_t_part, np.array(np.matmul(A_part,w) - b_part)) / b_size

def SGD_MB(A, b, x_0, r, alpha, iter_num, b_size):

    x_i = [x_0]
    y_s = [x_i[-1]]

    for k in range(iter_num):
        eta = 2 / (alpha * (k+1))
        ch_num = A.shape[0]/b_size
        if(ch_num - np.floor(ch_num)>0):
            ch_num = int(np.floor(ch_num)+1)
        else:
            ch_num = int(np.floor(ch_num))
        for s in range(ch_num):
            new_point = y_s[-1] - eta * SGD_MB_SO_f(A,b,y_s[-1], b_size)
            if (norm(new_point) > r):
                y_s.append(new_point / norm(new_point) * r)
            else:
                y_s.append(new_point)
        x_new = 0
        for k, y_k in enumerate(y_s):
            x_new += 2 * (k+1) / (ch_num * (ch_num+1)) * y_k
        x_i.append(x_new)
        y_s = [y_s[-1]]
    return x_i

def SVRG(A, b, x_0, r, alpha, beta, iter_num):

    A_t = np.transpose(A)

    k = 20 * (beta / alpha) # by lecture 8
    if (k - np.floor(k) > 0):
        k = int(np.floor(k) + 1)
    else:
        k = int(np.floor(k))

    eta = 1 / (10 * beta) # by lecture 8

    y_s = [x_0]
    for s in range(iter_num):
        x_s = [y_s[-1]]
        nabla_F = np.matmul(A_t, np.matmul(A, y_s[-1])-b)
        nabla_F = (1 / A.shape[0]) * nabla_F
        for t in range(k):
            # draw random gradient estimate index
            index = randint(A.shape[0])
            A_idx = A[index]
            A_idx_t = np.transpose(A_idx)
            b_idx = b[index]
            nabla_f_x = np.dot(A_idx_t, np.matmul(A_idx, x_s[-1]) - b_idx)
            nabla_f_y = np.dot(A_idx_t, np.matmul(A_idx, y_s[-1]) - b_idx)
            new_point = x_s[-1] - eta * (nabla_f_x - nabla_f_y + nabla_F)
            if (norm(new_point) > r):
                x_s.append(new_point / norm(new_point) * r)
            else:
                x_s.append(new_point)
        y_s.append(np.mean(x_s, axis=0))
    return y_s

###########################
######### R E S T #########
###########################

def create_matrix_A(m,n, s_min, s_max):
    A = np.random.randn(m, n)
    U,S,V = svd(A)
    s_min_ = S[-1]
    s_max_ = S[0]
    S = s_min + (S-s_min_)/(s_max_ - s_min_)*(s_max - s_min)
    A = np.matmul(np.matmul(U, diagsvd(S, m,n)), V)
    return  A

def data_creation(m,n,s_min, s_max):

    A = create_matrix_A(m,n, s_min, s_max)

    x_star = np.random.randn(n)
    x_star = x_star / np.linalg.norm(x_star)
    x_star *= np.random.uniform(0, 1)

    b = np.matmul(A, x_star) + 0.01 * np.random.randn(m)

    return A, x_star, b

def run_all(b_sizes):
    m = 250
    n = 15
    s_min = 0.5
    s_max = 4
    rad = 10

    A, x_star, b = data_creation(m, n, s_min, s_max)
    A_t = np.transpose(A)

    _, s, __ = svd(A)
    alpha = s[-1] ** 2
    beta = s[0] ** 2
    if (s[-1] ** 2 < 0.0000001):
        x_0 = np.random.randn(n)
        x_0 = (x_0 / norm(x_0)) * np.random.uniform(0, 5)
        X_n = GD(A, b, x_0, r=5, eta=1 / beta, iter_num=6000)
        base_error = 1 / 2 * (norm(np.matmul(A, X_n) - b) ** 2)
    else:
        A_tA = inv(np.matmul(A_t, A))
        sol_ = np.matmul(np.matmul(A_tA, A_t), b)
        base_error = 1 / 2 * (norm(np.matmul(A, sol_) - b) ** 2)

    x = np.random.randn(n)
    x = (x / norm(x)) * np.random.uniform(0, rad)

    SGD_iterations = SGD(A, b, x, rad, alpha, 600)
    SGD_res = []
    for x in SGD_iterations:
        SGD_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)

    SGD_MB_iterations = {}
    for b_size in b_sizes:
        SGD_MB_iterations[b_size] = SGD_MB(A, b, x, rad, alpha, 600, b_size)
    SGD_MB_res = {}
    for b_size in b_sizes:
        SGD_MB_res[b_size] = []
        for x in SGD_MB_iterations[b_size]:
            SGD_MB_res[b_size].append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)

    SVRG_iterations = SVRG(A, b, x, rad, alpha, beta, 600)
    SVRG_res = []
    for x in SVRG_iterations:
        SVRG_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)
    res = [SGD_res, SGD_MB_res, SVRG_res]
    return res


def plot_data(data, method_name, color):
    x = range(len(data[0]))
    averaged_data = np.sum(data, axis=0) / (np.sum(data != 0, axis=0))
    print(f'{method_name} : {averaged_data}')
    p = plt.plot(x[1:], averaged_data[1:], label=method_name, c=color)


if __name__ == '__main__':
    b_sizes = [2, 5, 10, 50]
    load_data = True
    if(load_data):
        SGD_file = open('q5_res/SGD.pkl', 'rb')
        SGD_res = pkl.load(SGD_file)
        SGD_file.close()

        SGD_MB_file = open('q5_res/SGD_MB.pkl', 'rb')
        SGD_MB_res = pkl.load(SGD_MB_file)
        SGD_MB_file.close()

        SVRG_file = open('q5_res/SVRG.pkl', 'rb')
        SVRG_res = pkl.load(SVRG_file)
        SVRG_file.close()

    else:
        res = []
        for k in range(20):
            res.append(run_all(b_sizes))
            print(f"Iteration {k+1} was implemented successfully")
        # print(f'res: {res}')

        SGD_res = [el[0] for el in res]
        SGD_file = open('q5_res/SGD.pkl', 'wb')
        pkl.dump(SGD_res, SGD_file, -1)
        SGD_file.close()

        SGD_MB_res_unpacked = [el[1] for el in res]
        SGD_MB_res = {b_size: [] for b_size in b_sizes}
        for run in SGD_MB_res_unpacked:
            for b_size in b_sizes:
                SGD_MB_res[b_size].append(run[b_size])

        SGD_MB_file = open('q5_res/SGD_MB.pkl', 'wb')
        pkl.dump(SGD_MB_res, SGD_MB_file, -1)
        SGD_MB_file.close()

        SVRG_res = [el[2] for el in res]
        SVRG_file = open('q5_res/SVRG.pkl', 'wb')
        pkl.dump(SVRG_res, SVRG_file, -1)
        SVRG_file.close()



    plot_data(SGD_res, f'SGD', color='r')

    plot_data(SGD_MB_res[2], f'SGD_MB, batch size = {2}', color='blue')
    plot_data(SGD_MB_res[5], f'SGD_MB, batch size = {5}', color='g')
    plot_data(SGD_MB_res[10], f'SGD_MB, batch size = {10}', color='y')
    # plot_data(SGD_MB_res[50], f'SGD_MB, batch size = {50}', color='purple')

    plot_data(SVRG_res, f'SVRG', color='black')

    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.show()