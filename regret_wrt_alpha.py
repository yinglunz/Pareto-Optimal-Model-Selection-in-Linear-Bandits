import numpy as np
import multiprocessing
import pickle
import time
from regret_curve import beta_list, algs, generate_problem, expressiveness, d, d_star, K, T, n_sims
from linear_class import LinUCB, LinUCBPlus, DynamicBalancing

def single_run(alg_obj):
    while alg_obj.t <= alg_obj.T:
        alg_obj.update()
        regret = alg_obj.get_data()['regret']
    return regret

def single_sim(algs, d, d_star, K, T):
    np.random.seed()
    the_problem = generate_problem(d, d_star, K, T)
    # different problem for each sim
    result = []
    for alg in algs:
        alg_obj = eval(alg)
        result.append(single_run(alg_obj))
    return result

def multi_sim(algs, d, d_star_list, K, T, ind):
    # to include different alphas
    result = {}
    for alg in algs:
        result[alg] = []
    for d_star in d_star_list:
        print('start with d_star = ', d_star)
        sim_data = single_sim(algs, d, d_star, K, T)
        for i in range(len(algs)):
            result[algs[i]].append(sim_data[i])
    return result

if __name__ == '__main__':
    np.random.seed()
    time_start = time.time()
    print('n_sims', n_sims)
    d_star_list = [5, 10, 15, 20, 25, 30, 35]
    alpha_list = [round(np.log(d_star)/np.log(T), 2) for d_star in d_star_list]
    the_problem = generate_problem(d, d_star, K, T)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    iters = [pool.apply_async(multi_sim, args=(algs, d, d_star_list, K, T, ind)) \
        for ind in range(n_sims)]
    results = {}
    results['setting_example'] = the_problem
    results['alpha_list'] = alpha_list
    results['d_star_list'] = d_star_list
    results['algs_label'] = {}
    results['regret_list'] = []
    for alg in algs:
        alg_obj = eval(alg)
        results['algs_label'][alg] = str(alg_obj)
    for result in iters:
        results['regret_list'].append(result.get())
    time_end = time.time()
    print('time consumed is !!', time_end-time_start)
    results['running_time'] = time_end - time_start
    with open('regret_wrt_alpha.dat', 'wb') as f:
        pickle.dump(results, f) 
    


  