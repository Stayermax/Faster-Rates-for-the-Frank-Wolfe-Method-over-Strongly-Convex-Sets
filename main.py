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

def CGD(A, b, x_0, r, iter_num):
    x_i = [x_0]
    A_t = np.transpose(A)

    for i in range(iter_num):
        eta = 2 / (2+i)
        nabla = np.matmul(A_t, np.matmul(A, x_i[-1])-b)
        v_t = -nabla/norm(nabla)*r
        new_point = x_i[-1] + eta*(v_t - x_i[-1])
        x_i.append(new_point)
        if norm(nabla) <= 0.00001:
            break
    return x_i

def CGD_Schatten_2(A, b, x_0, r, iter_num):
    x_i = [x_0]
    A_t = np.transpose(A)

    for i in range(iter_num):
        eta = 2 / (2+i)
        nabla = np.matmul(A_t, np.matmul(A, x_i[-1])-b)
        v_t = -nabla/norm(nabla)*r
        new_point = x_i[-1] + eta*(v_t - x_i[-1])
        x_i.append(new_point)
        if norm(nabla) <= 0.00001:
            break
    return x_i

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

def AGM(A, b, x_0, beta, delta_x=0.0005, epsilon=0.00001, iter_num=100):
    x = x_0
    x_i = [x_0]
    A_t = np.transpose(A)
    assert delta_x > 0, "Step must be positive."

    if (epsilon is None):
        epsilon = 0.05

    lambda_prev = 0
    lambda_curr = 1
    gamma = 1
    y_prev = x
    # beta = 0.05 / (2 * L)

    gradient = np.matmul(A_t, np.matmul(A, x_i[-1])-b)

    for i in range(iter_num):
        # y_curr = x - beta * gradient
        y_curr = x - ( 1 / beta ) * gradient
        x = (1 - gamma) * y_curr + gamma * y_prev
        x_i.append(x)
        y_prev = y_curr

        lambda_tmp = lambda_curr
        lambda_curr = (1 + np.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
        lambda_prev = lambda_tmp

        gamma = (1 - lambda_prev) / lambda_curr

        gradient = np.matmul(A_t, np.matmul(A, x_i[-1])-b)
        if norm(gradient) <= 0.00001:
            break

    return x_i

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

def run_all_methods(methods, iter_num = 100, mode = 'base_error_diff'):
    m = 250
    n = 15
    s_min = 0.5
    s_max = 4
    rad = 10

    # function creation
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

    res = {}

    CGD_lin_bound = 2 * beta * (2 * rad) ** 2
    alpha_K = 1 / rad
    M = alpha ** 0.5 * alpha_K / (8 * np.sqrt(2) * beta)
    CGD_sq_bound = max(9 / 2 * beta * ((2 * rad) ** 2), 18 * (M ** (-2)))
    L = 0.5

    if('CGD' in methods or 'all' in methods):
        CGD_res = []
        CGD_iterations = CGD(A, b, x, r=rad, iter_num=iter_num)
        for x in CGD_iterations:
            if(mode == 'with_diff'):
                CGD_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)
            else:
                CGD_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2))

        res['CGD_lin'] = [CGD_lin_bound / (i + 1) for i in range(iter_num)]
        res['CGD_sq'] = [CGD_sq_bound / ((i + 2) ** 2) for i in range(iter_num)]
        res['CGD'] = CGD_res

    if ('AGM' in methods or 'all' in methods):
        AGM_res = []
        AGM_iterations = AGM(A, b, x_0=x, beta=beta, delta_x=0.0005, iter_num=iter_num)
        for x in AGM_iterations:
            if(mode == 'with_diff'):
                AGM_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)
            else:
                AGM_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2))
        res['AGM'] = AGM_res

    # SGD
    if ('SGD' in methods or 'all' in methods):
        SGD_res = []
        SGD_iterations = SGD(A, b, x, rad, alpha, iter_num)
        for x in SGD_iterations:
            if(mode == 'with_diff'):
                SGD_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)
            else:
                SGD_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2))

        res['SGD'] = SGD_res

    # SVRG:
    if ('SVRG' in methods or 'all' in methods):
        SVRG_res = []
        SVRG_iterations = SVRG(A, b, x, rad, alpha, beta, iter_num)
        for x in SVRG_iterations:
            if(mode == 'with_diff'):
                SVRG_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)
            else:
                SVRG_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2))
        res['SVRG'] = SVRG_res
    return res

def run_all_Schatten_2(iter_num = 100):
    m = 250
    n = 15
    s_min = 0.5
    s_max = 4
    rad = 10

    # function creation
    A, x_star, b = data_creation(m, n, s_min, s_max)
    A_t = np.transpose(A)

    _, s, __ = svd(A)
    alpha = s[-1] ** 2
    beta = s[0] ** 2

    CGD_lin_bound = 2*beta*(2*rad)**2
    alpha_K = 1/rad

    CGD_iterations = CGD_Schatten_2(A, r=rad, iter_num=iter_num)
    CGD_Schatten_2_res = []
    for x in CGD_iterations:
        CGD_res.append(1 / 2 * (norm(np.matmul(A, x) - b) ** 2) - base_error)


    res = {'Schatten_2_res':CGD_Schatten_2_res,
           # 'CGD_sq':[CGD_sq_bound/((i+2)**2) for i in range(iter_num)],
           }
    return res

def average_data_calculation(data):
    max_dim = max(len(el) for el in data)
    sum_, num_ = [0]*max_dim, [0]*max_dim
    for el in data:
        for i in range(len(el)):
            if(el[i]!=0):
                sum_[i] += el[i]
                num_[i] += 1
    averaged_data = []
    for i in range(max_dim):
        if(num_[i]>0):
            averaged_data.append(sum_[i]/num_[i])
    return averaged_data

def plot_data(data, method_name, color, linestyle=None):
    x = range(max(len(el) for el in data))
    # averaged_data = np.sum(data, axis=0) / (np.sum(data != 0, axis=0))
    averaged_data = average_data_calculation(data)
    print(f'{method_name} : {averaged_data}')
    p = plt.plot(x[1:], averaged_data[1:], label=method_name, c=color, linestyle = linestyle)

def l_2(iter_num=600, methods=['all'], load_data=True, simulations_num=20):
    """

    Algorithms comparsion for l2 balls

    :param iter_num: Algorithm iterations
    :param methods:  ['CGD', 'AGD', 'SGD', 'SVRG'] or ['all']
    :param load_data: recalculate simulation if False, load saved simulation results if True
    :param simulations_num: Simulation runs
    """
    if(load_data):
        # Linear convergence rate
        CGD_lin_file = open(f'results/l_2/CGD_lin_{mode}.pkl', 'rb')
        CGD_lin_res = pkl.load(CGD_lin_file)
        CGD_lin_file.close()

        # Squared convergence rate
        CGD_sq_file = open(f'results/l_2/CGD_sq_{mode}.pkl', 'rb')
        CGD_sq_res = pkl.load(CGD_sq_file)
        CGD_sq_file.close()

        CGD_file = open(f'results/l_2/CGD_{mode}.pkl', 'rb')
        CGD_res = pkl.load(CGD_file)
        CGD_file.close()


        AGM_file = open(f'results/l_2/AGM_{mode}.pkl', 'rb')
        AGM_res = pkl.load(AGM_file)
        AGM_file.close()

        SGD_file = open(f'results/l_2/SGD_{mode}.pkl', 'rb')
        SGD_res = pkl.load(SGD_file)
        SGD_file.close()

        SVRG_file = open(f'results/l_2/SVRG_{mode}.pkl', 'rb')
        SVRG_res = pkl.load(SVRG_file)
        SVRG_file.close()
    else:
        res = []
        for k in range(simulations_num):
            res.append(run_all_methods(methods, iter_num, mode))
            # res.append(run_all_Schatten_2(iter_num))
            print(f"Iteration {k+1} was implemented successfully")

        # 1 Conditional gradient descent
        if('CGD' in methods or 'all' in methods):
            # Linear convergence rate
            CGD_lin_res = [res[0]['CGD_lin']]
            CGD_lin_file = open(f'results/l_2/CGD_lin_{mode}.pkl', 'wb')
            pkl.dump(CGD_lin_res, CGD_lin_file, -1)
            CGD_lin_file.close()

            # Squared convergence rate
            CGD_sq_res = [res[0]['CGD_sq']]
            CGD_sq_file = open(f'results/l_2/CGD_sq_{mode}.pkl', 'wb')
            pkl.dump(CGD_sq_res, CGD_sq_file, -1)
            CGD_sq_file.close()


            CGD_res = [el['CGD'] for el in res]
            CGD_file = open(f'results/l_2/CGD_{mode}.pkl', 'wb')
            pkl.dump(CGD_res, CGD_file, -1)
            CGD_file.close()
        # 2 Accelerated Nesterov Gradient Method
        if ('AGM' in methods or 'all' in methods):
            AGM_res = [el['AGM'] for el in res]
            AGM_file = open(f'results/l_2/AGM_{mode}.pkl', 'wb')
            pkl.dump(AGM_res, AGM_file, -1)
            AGM_file.close()
        # 3 Stochastic gradient descent
        if ('SGD' in methods or 'all' in methods):
            SGD_res = [el['SGD'] for el in res]
            SGD_file = open(f'results/l_2/SGD_{mode}.pkl', 'wb')
            pkl.dump(SGD_res, SGD_file, -1)
            SGD_file.close()
        # 4 Stochastic variance reduced gradient
        if ('SVRG' in methods or 'all' in methods):
            SVRG_res = [el['SVRG'] for el in res]
            SVRG_file = open(f'results/l_2/SVRG_{mode}.pkl', 'wb')
            pkl.dump(SVRG_res, SVRG_file, -1)
            SVRG_file.close()

    if ('CGD' in methods or 'all' in methods):
        plot_data(CGD_lin_res, f'CGD linear rate', color='black', linestyle='dashed')
        plot_data(CGD_sq_res, f'CGD square rate', color='pink', linestyle='dashed')
        plot_data(CGD_res, f'CGD', color='g')
    if ('AGM' in methods or 'all' in methods):
        plot_data(AGM_res, f'AGD', color='b')
        # pass
    if ('SGD' in methods or 'all' in methods):
        plot_data(SGD_res, f'SGD', color='r')
    if ('SVRG' in methods or 'all' in methods):
        plot_data(SVRG_res, f'SVRG', color='cyan')

    axes = plt.gca()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale('log')
    # axes.set_ylim([1, 1000])
    plt.legend()
    plt.show()

def Schatten_2(iter_num=600, methods=['all'], load_data=True, simulations_num=20):
    """

    Algorithms comparsion for Schatten2 balls

    :param iter_num: Algorithm iterations
    :param methods:  ['CGD', 'AGD', 'SGD', 'SVRG'] or ['all']
    :param load_data: recalculate simulation if False, load saved simulation results if True
    :param simulations_num: Simulation runs
    """
    if(load_data):
        # Linear convergence rate
        CGD_lin_file = open(f'results/Schatten_2/CGD_lin_{mode}.pkl', 'rb')
        CGD_lin_res = pkl.load(CGD_lin_file)
        CGD_lin_file.close()

        # Squared convergence rate
        CGD_sq_file = open(f'results/Schatten_2/CGD_sq_{mode}.pkl', 'rb')
        CGD_sq_res = pkl.load(CGD_sq_file)
        CGD_sq_file.close()

        CGD_file = open(f'results/Schatten_2/CGD_{mode}.pkl', 'rb')
        CGD_res = pkl.load(CGD_file)
        CGD_file.close()


        AGM_file = open(f'results/Schatten_2/AGM_{mode}.pkl', 'rb')
        AGM_res = pkl.load(AGM_file)
        AGM_file.close()

        SGD_file = open(f'results/Schatten_2/SGD_{mode}.pkl', 'rb')
        SGD_res = pkl.load(SGD_file)
        SGD_file.close()

        SVRG_file = open(f'results/Schatten_2/SVRG_{mode}.pkl', 'rb')
        SVRG_res = pkl.load(SVRG_file)
        SVRG_file.close()
    else:
        res = []
        for k in range(20):
            res.append(run_all_Schatten_2(iter_num))
            print(f"Iteration {k+1} was implemented successfully")

        # 1 Conditional gradient descent
        if('CGD' in methods or 'all' in methods):
            # Linear convergence rate
            CGD_lin_res = [res[0]['CGD_lin']]
            CGD_lin_file = open(f'results/Schatten_2/CGD_lin_{mode}.pkl', 'wb')
            pkl.dump(CGD_lin_res, CGD_lin_file, -1)
            CGD_lin_file.close()

            # Squared convergence rate
            CGD_sq_res = [res[0]['CGD_sq']]
            CGD_sq_file = open(f'results/Schatten_2/CGD_sq_{mode}.pkl', 'wb')
            pkl.dump(CGD_sq_res, CGD_sq_file, -1)
            CGD_sq_file.close()


            CGD_res = [el['CGD'] for el in res]
            CGD_file = open(f'results/Schatten_2/CGD_{mode}.pkl', 'wb')
            pkl.dump(CGD_res, CGD_file, -1)
            CGD_file.close()
        # 2 Accelerated Nesterov Gradient Method
        if ('AGM' in methods or 'all' in methods):
            AGM_res = [el['AGM'] for el in res]
            AGM_file = open(f'results/Schatten_2/AGM_{mode}.pkl', 'wb')
            pkl.dump(AGM_res, AGM_file, -1)
            AGM_file.close()
        # 3 Stochastic gradient descent
        if ('SGD' in methods or 'all' in methods):
            SGD_res = [el['SGD'] for el in res]
            SGD_file = open(f'results/Schatten_2/SGD_{mode}.pkl', 'wb')
            pkl.dump(SGD_res, SGD_file, -1)
            SGD_file.close()
        # 4 Stochastic variance reduced gradient
        if ('SVRG' in methods or 'all' in methods):
            SVRG_res = [el['SVRG'] for el in res]
            SVRG_file = open(f'results/Schatten_2/SVRG_{mode}.pkl', 'wb')
            pkl.dump(SVRG_res, SVRG_file, -1)
            SVRG_file.close()

    if ('CGD' in methods or 'all' in methods):
        plot_data(CGD_lin_res, f'CGD linear rate', color='black', linestyle='dashed')
        plot_data(CGD_sq_res, f'CGD square rate', color='pink', linestyle='dashed')
        plot_data(CGD_res, f'CGD', color='g')
    if ('AGM' in methods or 'all' in methods):
        plot_data(AGM_res, f'AGD', color='b')
        # pass
    if ('SGD' in methods or 'all' in methods):
        plot_data(SGD_res, f'SGD', color='r')
    if ('SVRG' in methods or 'all' in methods):
        plot_data(SVRG_res, f'SVRG', color='cyan')

    axes = plt.gca()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale('log')
    # axes.set_ylim([1, 1000])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    iter_num = 600
    # methods = ['CGD']
    # methods = ['CGD', 'AGD', 'SGD', 'SVRG']
    methods = ['all']
    mode = 'with_diff'
    # mode = 'no_diff'
    load_data = False
    simulations_num = 20

    # l_2(iter_num, methods, load_data, simulations_num)
    Schatten_2(iter_num, methods, load_data, simulations_num)