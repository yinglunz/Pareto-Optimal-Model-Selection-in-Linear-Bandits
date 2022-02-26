import numpy as np
import multiprocessing
import pickle
import time
import matplotlib.pyplot as plt
import copy

n_data_points = 100
# number of data points appearing on the plot, interval = T / n_data_points 
beta_list = [0.5]
expressiveness = False
d = 600
d_star = 12
K = 1200
T = 2500
n_sims = 100
# number of simulation
algs = ['LinUCB(the_problem, False, {})'.format(expressiveness), \
    'LinUCB(the_problem, True, {})'.format(expressiveness)] + \
    ['DynamicBalancing(the_problem, {})'.format(expressiveness)] + \
    ['LinUCBPlus(the_problem, {}, {})'.format(beta, expressiveness) for beta in beta_list] 

from linear_class import LinUCB, LinUCBPlus, DynamicBalancing

def make_expressive(action_set, d, d_star, K, T, wrt_beta=True):
    if wrt_beta == True:
        # make action expressive wrt LinUCB++
        l = min(int(np.ceil(np.log2(T**max(beta_list)))) + 1, int(np.floor(np.log2(d))))
    else:
        # make action expressive for all 2**i \leq d
        l = min(int(np.ceil(np.log2(T))), int(np.floor(np.log2(d))))
    start_di = 3
    di_list = [2**i for i in range(l+1)]
    di_list = [2**i for i in range(start_di, l+1)]
    action_set_di = copy.deepcopy(action_set)
    action_set_expressive = np.hstack((action_set_di[:,:di_list[0]], np.zeros((K, d-di_list[0]))))
    for i in range(1, len(di_list)):
        action_set_di = copy.deepcopy(action_set)
        action_set_di = np.hstack((action_set_di[:,:di_list[i]], np.zeros((K, d-di_list[i]))))
        action_set_expressive = np.vstack((action_set_expressive, action_set_di))
    action_set_di = copy.deepcopy(action_set)
    action_set_expressive = np.vstack((action_set_expressive, action_set_di))
    di_list.append(d)
    # append the full dimension d  
    action_set_expressive_oracle = copy.deepcopy(action_set)
    action_set_expressive_oracle = action_set_expressive_oracle[:,:d_star]
    l_oracle = int(np.log2(d_star))
    if start_di <= l_oracle:
        di_list_oracle = [2**i for i in range(start_di, l_oracle+1)]
        for i in range(len(di_list_oracle)):
            action_set_di_oracle = copy.deepcopy(action_set)[:, :d_star]
            action_set_di_oracle = np.hstack((action_set_di[:,:di_list_oracle[i]], np.zeros((K, d_star-di_list_oracle[i]))))
            action_set_expressive_oracle = np.vstack((action_set_expressive_oracle, action_set_di_oracle))
    return  action_set_expressive, action_set_expressive_oracle, di_list

def generate_problem(d, d_star, K, T):
    theta_star_oracle = np.ones(d_star)
    # theta_star_oracle = 1/np.sqrt(np.arange(1, d_star+1))
    # theta_star_oracle = 1/np.sqrt(np.arange(1, d_star+1))[::-1]
    # above two theta_stars are used in to generate plots in Appendix D
    theta_star_oracle /= np.linalg.norm(theta_star_oracle)
    theta_star = np.hstack((theta_star_oracle, np.zeros(d-d_star)))
    theta_star_oracle = theta_star[:d_star]
    action_set = np.random.randn(d, K)
    action_set /= np.linalg.norm(action_set, axis=0)
    action_set_origin = action_set.transpose()
    true_reward_origin = np.matmul(action_set_origin, theta_star)
    opt_reward_origin = max(true_reward_origin)
    min_reward_origin = min(true_reward_origin)
    action_set_expressive, action_set_expressive_oracle, di_list_final = \
        make_expressive(action_set_origin, d, d_star, K, T, True)
    true_reward_expressive = np.matmul(action_set_expressive, theta_star)
    opt_reward_expressive = max(true_reward_expressive)
    min_reward_expressive = min(true_reward_expressive)
    true_reward_expressive_oracle = np.matmul(action_set_expressive_oracle, theta_star_oracle)
    opt_reward_expressive_oracle = max(true_reward_expressive_oracle)
    min_reward_expressive_oracle = min(true_reward_expressive_oracle)
    print('problem instance generated, expressive = ', expressiveness)
    the_problem = {'d': d,
                   'd_star': d_star,
                   'T': T,
                   'K': K,
                   'alpha': np.log(d_star)/np.log(T),
                   # alpha represents the hardness level
                   'theta_star_oracle': theta_star_oracle,
                   # theta_star truncated at d_star
                   'theta_star': theta_star,
                   'action_set_origin': action_set_origin,
                   'true_reward_origin': true_reward_origin,
                   'action_set_expressive': action_set_expressive,
                   'true_reward_expressive': true_reward_expressive,
                   'action_set_expressive_oracle': action_set_expressive_oracle,
                   'true_reward_expressive_oracle': true_reward_expressive_oracle,
                   'opt_reward_origin': opt_reward_origin,
                   'min_reward_origin': min_reward_origin,
                   'opt_reward_expressive': opt_reward_expressive,
                   'min_reward_expressive': min_reward_expressive,
                   'opt_reward_expressive_oracle': opt_reward_expressive_oracle,
                   'min_reward_expressive_oracle': min_reward_expressive_oracle,
                   'stddev': 0.1,
                   'lambda_para': 0.1,
                   # regularization coefficient in ridge regression
                   'expressive': expressiveness,
                   'di_list':di_list_final
                   }
    return the_problem

def single_run(alg_obj):
    interval = alg_obj.T // n_data_points
    if interval%2 == 1:
        interval += 1
    # we make the interval even since Smooth Corral take two steps each time
    sim_data = {}
    while alg_obj.t <= alg_obj.T:
        alg_obj.update()
        if alg_obj.t % interval == 0:
            data_obj = alg_obj.get_data()
            sim_data[alg_obj.t] = data_obj    
    return sim_data

def single_sim(algs, d, d_star, K, T, ind):
    np.random.seed()
    result = {}
    the_problem = generate_problem(d, d_star, K, T)
    for alg in algs:
        alg_obj = eval(alg)
        result[alg] = single_run(alg_obj)
    return result

if __name__ == '__main__':
    np.random.seed()
    time_start = time.time()
    the_problem = generate_problem(d, d_star, K, T)
    results = {}
    results['setting_example'] = the_problem
    results['algs_label'] = {}
    for alg in algs:
        alg_obj = eval(alg)
        results['algs_label'][alg] = str(alg_obj)
    results['regret_list'] = []
    pool = multiprocessing.Pool(processes=2*multiprocessing.cpu_count())
    iters = [pool.apply_async(single_sim, args=(algs, d, d_star, K, T, ind)) \
        for ind in range(n_sims)]
    for result in iters:
        results['regret_list'].append(result.get()) 
    time_end = time.time()
    results['running_time'] = time_end - time_start
    with open('regret_curve.dat', 'wb') as f:
        pickle.dump(results, f)
    print('total time consumed is ', time_end-time_start)



    


    
    
