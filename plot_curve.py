import numpy as np
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from regret_curve import algs
line_style_list = [(0, (1, 1)), '-', (0, (5, 5)), (0, (5, 1)), '-.', (0, (3, 1, 1, 1, 1, 1))]
marker_list = ['X', 's', '^', 'h', 'P', '*']
font_size = 10
legend_size = 10

if __name__ == '__main__':
    with open('regret_curve.dat', 'rb') as f:
        x = pickle.load(f)
    running_time = x['running_time']
    print('total running time is ', running_time)
    the_problem = x['setting_example']
    sim_data = x['regret_list']
    n_sims = len(sim_data)
    print('n_sim = ', n_sims)
    expressiveness = the_problem['expressive']
    print('expressive =', expressiveness)
    algs_label = x['algs_label']
    alpha = the_problem['alpha']
    d = the_problem['d']
    d_star = the_problem['d_star']
    K = the_problem['K']
    T = the_problem['T']
    if expressiveness == False:
        opt_reward = round(the_problem['opt_reward_origin'], 2)
    else:
        opt_reward = round(the_problem['opt_reward_expressive'], 2)
    stddev = the_problem['stddev']
    times = list(sim_data[0][algs[0]].keys())
    regret_curve = {}
    for alg in algs:
        regret_curve[alg] = np.array([[0. for time in times] for sim in range(n_sims)])
    for sim in range(n_sims):
        sim_data = x['regret_list'][sim]
        for alg in sim_data:
            alg_data = sim_data[alg]
            for t_ind in range(len(times)):
                t = times[t_ind]
                regret_curve[alg][sim][t_ind] = alg_data[t]['regret']

    fig = plt.figure(); ax = fig.add_subplot(111)
    for i in range(len(algs)):
        ave = np.mean(regret_curve[algs[i]], axis=0)
        std = np.std(regret_curve[algs[i]], axis=0)
        print('{}-ave{}-std{}-ratio{}'.format(algs_label[algs[i]], ave[-1], std[-1], std[-1]/ave[-1]))
        plt.plot(times, ave, label=algs_label[algs[i]], linestyle = line_style_list[i], linewidth=3)
        plt.fill_between(times, ave-std, ave+std, alpha=0.2)
    ax.set_ylabel('Expected regret', fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)
    plt.legend(loc='best', prop={'size':legend_size})
    plt.grid(alpha=0.75)
    plt.savefig('regret_curve.pdf')
    plt.show()


