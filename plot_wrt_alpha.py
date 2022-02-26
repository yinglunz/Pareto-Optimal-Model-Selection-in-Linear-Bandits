import numpy as np
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from regret_curve import algs
from plot_curve import line_style_list, marker_list, font_size, legend_size

if __name__ == '__main__':
    with open('regret_wrt_alpha.dat', 'rb') as f:
        x = pickle.load(f)
    running_time = x['running_time']
    print('total running time is ', running_time)
    the_problem = x['setting_example']
    alpha_list = x['alpha_list']
    algs_label = x['algs_label']
    print(algs_label)
    d_star_list = x['d_star_list']
    print(algs)
    expressiveness = the_problem['expressive']
    print('expressive =', expressiveness)
    d = the_problem['d']
    d_star = the_problem['d_star']
    K = the_problem['K']
    if expressiveness == False:
        opt_reward = round(the_problem['opt_reward_origin'], 2)
    else:
        opt_reward = round(the_problem['opt_reward_expressive'], 2)
    sim_data = x['regret_list']
    n_sims = len(sim_data)
    print('n_sims = ', n_sims)
    regret_wrt_alpha = {}
    for alg in algs:
        regret_wrt_alpha[alg] = np.array([[0. for alpha in alpha_list] for sim in range(n_sims)])
    for sim in range(n_sims):
        sim_data = x['regret_list'][sim]
        for alg in sim_data:
            alg_data = sim_data[alg]
            for i in range(len(alpha_list)):
                regret_wrt_alpha[alg][sim][i] = alg_data[i]

    fig = plt.figure(); ax = fig.add_subplot(111)
    for i in range(len(algs)):
        ave = np.mean(regret_wrt_alpha[algs[i]], axis=0)
        std = np.std(regret_wrt_alpha[algs[i]], axis=0)
        plt.plot(alpha_list, ave, label=algs_label[algs[i]], \
            marker = marker_list[i], linestyle = line_style_list[i], linewidth=3, markersize=9)
        plt.fill_between(alpha_list, ave-std, ave+std, alpha=0.2)
    ax.set_ylabel('Expected regret', fontsize = font_size)
    ax.set_xlabel(r'$\alpha$', fontsize = font_size)
    plt.legend(loc='best', prop={'size':legend_size})
    plt.grid(alpha=0.75)
    plt.savefig('regret_wrt_alpha.pdf')
    plt.show()


