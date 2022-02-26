import numpy as np
from enum import Enum
from math import log, sqrt, exp
import copy

def compute_upper_bound(theta_hat, action_set, design_inv, gamma):
    emp_mean = np.matmul(action_set, theta_hat)
    exploration = np.einsum('ij,ji->i', np.dot(action_set, design_inv), action_set.T)
    return emp_mean + gamma * exploration

def update_inverse(design_inv, feature):
    # apply Sherman-Morrison formula to update design_inv, with u = v = feature
    return design_inv - np.outer(np.matmul(design_inv, feature), np.matmul(feature, design_inv))/ (1 + np.inner(feature, np.matmul(design_inv, feature)))

class LinUCB(object):
    def __init__(self, the_problem, oracle, expressive):
        self.oracle = oracle
        self.expressive = expressive
        if self.oracle == True:
            self.d = the_problem['d_star']
            self.theta_star = the_problem['theta_star_oracle']
            if self.expressive == False:
                self.action_set = the_problem['action_set_origin'][:,:self.d]
                self.true_reward = the_problem['true_reward_origin']
                self.opt_reward = the_problem['opt_reward_origin']
                self.K = len(self.action_set)
            else:
                self.action_set = the_problem['action_set_expressive_oracle']
                self.true_reward = the_problem['true_reward_expressive_oracle']
                self.opt_reward = max(self.true_reward)
                self.K = len(self.true_reward)
        else:
            self.d = the_problem['d']
            self.theta_star = the_problem['theta_star']
            if self.expressive == False:
                self.action_set = the_problem['action_set_origin']
                self.true_reward = the_problem['true_reward_origin']
                self.opt_reward = max(self.true_reward)
                self.K = len(self.true_reward)
            else:
                self.action_set = the_problem['action_set_expressive']
                self.true_reward = the_problem['true_reward_expressive']
                self.opt_reward = max(self.true_reward)
                self.K = len(self.true_reward)

        self.T = the_problem['T']
        self.stddev = the_problem['stddev']
        self.lambda_para = the_problem['lambda_para']
        self.gamma = self.stddev * np.sqrt(2*np.log(2*(self.T**1.5)*self.K))
        # gamma is the coefficient for the exploration term in LinUCB
        # we set confidence parameter delta = 1/sqrt(T)
        self.t = 0
        self.regret = 0
        self.pulls = np.zeros(self.K)
        self.design_matrix = self.lambda_para * np.identity(self.d)
        self.design_inv = np.linalg.inv(self.design_matrix)
        self.b_vector = np.zeros(self.d)
    
    def __str__(self):
        if self.oracle == True:
            return 'LinUCB Oracle'
        else:
            return 'LinUCB'

    def pull_arm(self, arm):
        return np.random.normal(np.inner(self.theta_star, self.action_set[arm]), self.stddev)
    
    def update(self):
        self.t += 1
        theta_hat = np.matmul(self.design_inv, self.b_vector)
        upper_bounds = list(compute_upper_bound(theta_hat, self.action_set, self.design_inv, self.gamma))
        max_value = max(upper_bounds)
        arm = np.random.choice([i for i, j in enumerate(upper_bounds) if j == max_value])
        self.pulls[arm] += 1
        reward = self.pull_arm(arm)
        self.design_matrix += np.outer(self.action_set[arm], self.action_set[arm])
        self.design_inv = update_inverse(self.design_inv, self.action_set[arm])
        self.b_vector += self.action_set[arm] * reward
        self.regret += self.opt_reward - self.true_reward[arm]
        if self.t % 500 == 0:
            print('LinUCB oracle={}, time = {}'.format(self.oracle, self.t))
            print('arm upper bound = {} emp mean = {}, true mean = {}'.format(max_value, np.dot(theta_hat, self.action_set[arm]), self.true_reward[arm]))
    
    def get_data(self):
        data = {'t': self.t, 'regret':self.regret}
        return data   

class LinUCBPlus(object):
    def __init__(self, the_problem, beta, expressive):
        self.beta = beta
        self.expressive = expressive
        self.d = the_problem['d']
        self.theta_star = the_problem['theta_star']
        self.T = the_problem['T']
        self.K_origin = the_problem['K']
        self.stddev = the_problem['stddev']
        self.lambda_para = the_problem['lambda_para']
        self.di_list = the_problem['di_list']

        if self.expressive == True:
            self.action_set = the_problem['action_set_expressive']
            self.true_reward = the_problem['true_reward_expressive']
            self.opt_reward = the_problem['opt_reward_expressive']
            self.min_reward = the_problem['min_reward_expressive']
            self.range_reward = self.opt_reward - self.min_reward
        else:
            self.action_set = the_problem['action_set_origin']
            self.true_reward = the_problem['true_reward_origin']
            self.opt_reward = the_problem['opt_reward_origin']
            self.min_reward = the_problem['min_reward_origin']
            self.range_reward = self.opt_reward - self.min_reward
        
        self.K = self.action_set.shape[0]
        self.p = int(np.ceil(np.log2(self.T**beta)))
        self.empirical_frequency = np.zeros((self.p, self.K+self.p-1))
        # row i (start from 0) represents the empirical sampling frequency at iteration (i+1);
        # the additional self.p - 1 columns are for virtual arms.
        self.true_reward_extra = np.zeros(self.p-1)
        # p-1 dimensional matrix that stores true reward of virtual actions
        self.t = 0
        self.regret = 0
        self.i = 0
    
    def __str__(self):
        return 'LinUCB++ (ours)'
  
    def pull_arm_regular(self, arm):
        return np.random.normal(self.true_reward[arm], self.stddev)
        
    def pull_arm_extra(self, arm):
        i = arm - self.K
        while i >= 0:
            arm = np.argmax(np.random.multinomial(1, self.empirical_frequency[i]))
            i = arm - self.K
        return self.pull_arm_regular(arm)
            
    def update(self):
        self.t += 1
        if self.t == (2**self.p) * (2**(self.i+1)-2) + 1:
            # time when we need to prepare for the next iteration
            self.i += 1
            self.di = max(min(2**(self.p + 2 - self.i), self.d),1)
            self.di_aug = self.di + self.i - 1
            # di_aug represented the augmented dimension
            self.Delta_Ti = min(2**(self.p + self.i), self.T)
            if self.expressive == True:
                self.K_eff_i = sum(i <= self.di for i in self.di_list) * self.K_origin
                if self.K_eff_i == 0:
                    self.K_eff_i = self.K_origin
            else:
                self.K_eff_i = self.K_origin 
            self.action_set_i = copy.deepcopy(self.action_set)[:self.K_eff_i, :self.di]
            self.Ki = self.K_eff_i + self.i - 1
            # Ki represents the total number of arms, including all virtual arms
            self.gamma_i = np.sqrt(2*(self.stddev**2)*np.log(2*(self.Delta_Ti**1.5)*self.Ki))
            self.gamma_i_virtual = np.sqrt(2*(self.range_reward**2/4+self.stddev**2)*np.log(2*(self.Delta_Ti**1.5)*self.Ki))
            # we set confidence parameter delta = 1/sqrt(T)
            # the (self.range_reward**2/4+self.stddev**2) term is for the square of sub-gaussianness parameter for the virtual arms. self.range_reward can be replaced by an upper bound of it.
            if self.i > 1:
                self.true_reward_extra[self.i-1-1] = \
                    np.inner(self.empirical_frequency[self.i-2][:self.K], self.true_reward)+ \
                    np.inner(self.empirical_frequency[self.i-2][self.K:], self.true_reward_extra)
                # we are currently in iteration i, prepare for virtual arm in iteration i-1
                # we have another -1 here since index starts from 0
            max_norm = max(np.linalg.norm(self.action_set_i, axis=1))
            # we make the norm of virtual arms equal to the largest norm among real arms
            # this doesn't affect the noisy rewards since they are generated based on true rewards, see function pull_arm_regular
            # this could affect the theta_star to be learned, but it can only decrease its norm since we increase the norms of either real or virtual arms
            if max_norm <= 1:
                self.action_set_i_aug = np.hstack( ( np.vstack((self.action_set_i / max_norm, np.zeros((self.i-1,self.di)))), np.vstack((np.zeros((self.K_eff_i, self.i-1)), np.identity(self.i-1))) ) )
            else:
                self.action_set_i_aug = np.hstack( ( np.vstack((self.action_set_i , np.zeros((self.i-1,self.di)))), np.vstack((np.zeros((self.K_eff_i, self.i-1)), max_norm * np.identity(self.i-1))) ) )
            # augmented action set at iteration i, dimension is Ki-by-di_aug
            self.design_matrix = self.lambda_para * np.identity(self.di_aug)
            self.design_inv = np.linalg.inv(self.design_matrix)
            self.b_vector = np.zeros(self.di_aug)
        
        theta_hat = np.matmul(self.design_inv, self.b_vector)
        upper_bounds = list(compute_upper_bound(theta_hat, self.action_set_i_aug[:self.K_eff_i,:], self.design_inv, self.gamma_i)) +\
             list(compute_upper_bound(theta_hat, self.action_set_i_aug[self.K_eff_i:,:], self.design_inv, self.gamma_i_virtual))
        max_value = max(upper_bounds)
        arm = np.random.choice([i for i, j in enumerate(upper_bounds) if j == max_value])
        self.design_matrix += np.outer(self.action_set_i_aug[arm], self.action_set_i_aug[arm])
        self.design_inv = update_inverse(self.design_inv, self.action_set_i_aug[arm])

        if arm < self.K_eff_i:
            # if we select a regular arm to pull
            self.empirical_frequency[self.i-1, arm] += 1/self.Delta_Ti
            emp_mean = np.matmul(theta_hat, self.action_set_i_aug[arm])
            true_mean = self.true_reward[arm] 
            self.regret += self.opt_reward - true_mean
            reward = self.pull_arm_regular(arm)        
        else:
            # otherwise we pull a virtual arm
            arm_true = arm - self.K_eff_i + self.K
            # true number of virtual arm
            self.empirical_frequency[self.i-1, arm_true] += 1/self.Delta_Ti
            emp_mean = np.matmul(theta_hat, self.action_set_i_aug[arm])
            true_mean = self.true_reward_extra[arm_true-self.K]
            self.regret += self.opt_reward - true_mean
            reward = self.pull_arm_extra(arm_true)
        self.b_vector += self.action_set_i_aug[arm] * reward
        
        if self.t % 500 == 0:
            print('LinUCB++, time = {} iteration = {}'.format(self.t, self.i))
            print('upper bound = {}, emp_mean = {}, true_mean = {}'.format(max_value, emp_mean, true_mean))
            print('frequency of pulling virtual arms = ', self.empirical_frequency[:, -self.p+1:])

    def get_data(self):
        data = {'t': self.t, 'regret': self.regret}
        return data

class LinUCB_base(object):
    def __init__(self, the_problem, d, T, expressive=False):
        self.expressive = expressive
        self.d = d
        self.K_origin = the_problem['K']
        if self.expressive == False:
            self.action_set = the_problem['action_set_origin'][:,:self.d]
            self.true_reward = the_problem['true_reward_origin']
            self.K = len(self.true_reward)
            self.opt_reward = max(self.true_reward)
        else:
            self.action_set = the_problem['action_set_expressive'][:,:self.d]
            self.true_reward = the_problem['true_reward_expressive']
            self.K = len(self.true_reward)
            self.opt_reward = max(self.true_reward)
            self.di_list = the_problem['di_list']
            if self.K_origin * len(self.di_list) != self.action_set.shape[0]:
                raise Exception('LinUCB_base incorrect calculation of numebr of effective arms')
            K_eff = sum(i <= self.d for i in self.di_list) * self.K_origin
            if K_eff == 0:
                K_eff = self.K_origin
            self.K = K_eff
            self.action_set = self.action_set[:self.K, :]

        self.T = T
        self.stddev = the_problem['stddev']
        self.lambda_para = the_problem['lambda_para']
        self.gamma = self.stddev * np.sqrt(2*np.log(2*(self.T**1.5)*self.K))
        # we set confidence parameter delta = 1/sqrt(T)
        self.t = 0
        self.regret = 0
        # regret suffered from self.update()
        self.regret_total = 0
        # regret suffered for both self.update() and self.dummy_update()
        self.pulls = np.zeros(self.K)
        self.design_matrix = self.lambda_para * np.identity(self.d)
        self.design_inv = np.linalg.inv(self.design_matrix)
        self.b_vector = np.zeros(self.d)
        self.theta_hat = np.matmul(self.design_inv, self.b_vector)
        self.upper_bounds = 10000* np.ones(self.K)
    
    def __str__(self):
        if self.expressive == True:
            return 'LinUCB_base_d={}_expressive'.format(self.d)
        else:
            return 'LinUCB_base_d={}_non_expressive'.format(self.d)

    def pull_arm(self, arm):
        return self.true_reward[arm] + np.random.normal(0, self.stddev)
    
    def dummy_update(self, upper_bounds):
        # for step 2 in the smoothed algorithm, the policy of base algorithm is not updated according to the design of the smoothed algorithm.
        max_value = max(upper_bounds)
        arm = np.random.choice([i for i, j in enumerate(upper_bounds) if j == max_value])
        reward = self.pull_arm(arm)
        instant_regret = self.opt_reward - self.true_reward[arm]
        self.regret_total += instant_regret
        return reward, instant_regret, arm
    
    def update(self):
        self.t += 1
        max_value = max(self.upper_bounds)
        arm = np.random.choice([i for i, j in enumerate(self.upper_bounds) if j == max_value])
        self.pulls[arm] += 1
        reward = self.pull_arm(arm)
        self.design_matrix += np.outer(self.action_set[arm], self.action_set[arm])
        self.design_inv = update_inverse(self.design_inv, self.action_set[arm])
        self.b_vector += self.action_set[arm] * reward
        self.theta_hat = np.matmul(self.design_inv, self.b_vector)
        self.upper_bounds = list(compute_upper_bound(self.theta_hat, self.action_set, self.design_inv, self.gamma))
        instant_regret = self.opt_reward - self.true_reward[arm]
        self.regret += instant_regret
        self.regret_total += instant_regret
        
        if self.t % 200 == 0:
            print('LinUCB_base+d={}, time = {}'.format(self.d, self.t))

        return reward, instant_regret, arm

    def get_data(self):
        data = {'t': self.t, 'regret': self.regret}
        return data

class UCB_base(object):
    def __init__(self, the_problem, virtual_arm, virtual_true_reward, T, p, expressive):
        # virtual_arm is a matrix of empirical sample frequency
        self.K = virtual_arm.shape[0]
        self.empirical_frequency = virtual_arm
        self.expressive = expressive
        self.theta_star = the_problem['theta_star']
        if self.expressive == False:
            self.action_set = the_problem['action_set_origin']
            self.true_reward = the_problem['true_reward_origin']
            self.opt_reward = max(self.true_reward)
        else:
            self.action_set = the_problem['action_set_expressive']
            self.true_reward = the_problem['true_reward_expressive']
            self.opt_reward = max(self.true_reward)
        self.true_reward_virtual = virtual_true_reward
        self.K_real = len(self.action_set)
        self.T = T
        self.p = p
        self.d = the_problem['d']
        self.stddev = the_problem['stddev']
        self.t = 0
        self.regret = 0
        # regret suffered from self.update()
        self.regret_total = 0
        # regret suffered for both self.update() and self.dummy_update()
        self.pulls = np.zeros(self.K)
        self.rewards_emp = np.zeros(self.K)
        self.means_emp = np.zeros(self.K)
        self.upper_bounds = 10000* np.ones(self.K)
    
    def __str__(self):
        if self.expressive == True:
            return 'UCB_base_expressive'
        else:
            return 'UCB_base_non_expressive'
    
    def pull_arm_regular(self, arm):
        return np.random.normal(self.true_reward[arm], self.stddev)

    def pull_arm_virtual(self, arm):
        i = arm - self.K_real
        while i >= 0:
            arm = np.argmax(np.random.multinomial(1, self.empirical_frequency[i]))
            i = arm - self.K_real
        return self.pull_arm_regular(arm)

    def pull_arm(self, arm):
        arm = arm + self.K_real
        return self.pull_arm_virtual(arm)
    
    def dummy_update(self, upper_bounds):
        # for step 2 in the smoothed algorithm, the policy of base algorithm is not updated according to the design of the smoothed algorithm.
        max_value = max(upper_bounds)
        arm = np.random.choice([i for i, j in enumerate(upper_bounds) if j == max_value])
        reward = self.pull_arm(arm)
        instant_regret = self.opt_reward - self.true_reward_virtual[arm]
        self.regret_total += instant_regret
        # the index of this virtual arm equals arm + self.K_real
        return reward, instant_regret, arm + self.K_real

    def update(self):
        self.t += 1
        max_value = max(self.upper_bounds)
        arm = np.random.choice([i for i, j in enumerate(self.upper_bounds) if j == max_value])
        self.pulls[arm] += 1
        reward = self.pull_arm(arm)
        self.rewards_emp[arm] += reward
        self.means_emp[arm] = self.rewards_emp[arm]/self.pulls[arm]
        n = self.pulls[arm]
        self.upper_bounds[arm] = self.means_emp[arm] + sqrt((1+n)/(n**2) * (1 + 4 * sqrt(self.T) * log(self.K * (1+n)**(1/2) ) ) ) 
        instant_regret = self.opt_reward - self.true_reward_virtual[arm]
        self.regret += instant_regret
        self.regret_total += instant_regret
        if self.t % 200 == 0:
            print('UCB_base, time = {}'.format(self.t))
        # the index of this virtual arm is arm + self.K_real
        return reward, instant_regret, arm + self.K_real

    def get_data(self):
        data = {'t': self.t, 'regret':self.regret}
        return data

class DynamicBalancing(object):
    def __str__(self):
        return 'Dynamic Balancing'
        # Our implementation of Dynamic Balancing 
        # http://proceedings.mlr.press/v139/cutkosky21a.html
    def __init__(self, the_problem, expressive=False):
        self.expressive = expressive
        self.d = the_problem['d']
        # the ambient dimension
        self.T = the_problem['T']
        self.delta = 1/sqrt(self.T)
        if expressive == False:
            self.opt_reward = the_problem['opt_reward_origin']
            self.min_reward = the_problem['min_reward_origin']
            self.true_reward = the_problem['true_reward_origin']
        else:
            self.opt_reward = the_problem['opt_reward_expressive']
            self.min_reward = the_problem['min_reward_expressive']
            self.true_reward = the_problem['true_reward_expressive']
        self.K = len(self.true_reward)
        self.M = int(np.ceil(np.log2(self.d)))
        # number of base learners
        self.d_list = [2**i for i in range(self.M)]
        self.base_algs = [LinUCB_base(the_problem, d, self.T, self.expressive) for d in self.d_list]
        self.t = 0
        self.total_regret = 0
        self.base_regrets = np.zeros(self.M)
        # cumulative regret for each base learner
        self.active_set = list(np.arange(self.M))
        self.base_regrets_predicted = np.zeros(self.M)
        self.base_rewards = np.zeros(self.M)
        self.base_counts = np.zeros(self.M)
        self.base_ave_rewards = np.zeros(self.M)
        self.base_confidence_bounds = np.zeros(self.M)
        self.c = 1
        # we set constant c = 1 for confidence bounds

    def sample_base(self):
        min_value = min(self.base_regrets_predicted[self.active_set])
        idx = np.random.choice([i for i, j in enumerate(self.base_regrets_predicted) if j == min_value] )
        return idx

    def get_regret_bound(self, idx):
        d = self.d_list[idx]
        t = self.base_counts[idx]
        K = self.K
        return max(sqrt(d * t * ( np.log(max(K * (t**1.5) * np.log(t), 1)) ) ** 3), 1)

    def get_v(self, idx):
        return 1

    def get_b(self, idx, t):
        return 0
        # v and b are set based on the choices mentioned in Section 7 of
        # http://proceedings.mlr.press/v139/cutkosky21a.html

    def update_active_set(self):
        upperCB = self.base_ave_rewards + self.base_confidence_bounds
        max_value = max(upperCB)
        self.active_set = []
        for i in range(self.M):
            if self.base_counts[i] == 0:
                self.active_set.append(i)
                # arms never pulled are added into the active set
            elif upperCB[i] + self.get_regret_bound(i)/self.base_counts[i] >= max_value:
                self.active_set.append(i)
    
    def update(self):
        base_idx = self.sample_base()
        self.t += 1
        reward, instant_regret, arm_set = self.base_algs[base_idx].update()
        self.total_regret += instant_regret
        self.base_rewards[base_idx] += reward
        self.base_counts[base_idx] += 1
        self.base_regrets_predicted[base_idx] = self.get_v(base_idx) * self.get_regret_bound(base_idx)
        self.base_ave_rewards[base_idx] = (self.base_rewards[base_idx] / self.base_counts[base_idx]) - self.get_b(base_idx, self.t)
        self.base_confidence_bounds[base_idx] = \
            self.c * sqrt( log( max(self.M * log(self.base_counts[base_idx]) / self.delta, 1) ) \
                / self.base_counts[base_idx] )
        self.update_active_set()
        self.base_regrets[base_idx] = self.base_algs[base_idx].regret_total

        if self.t % 500 == 0:
            print('Dynamic Balancing at time = {}'.format(self.t))
            print('base expected regrets = {}'.format(self.base_regrets))
            print('base counts = {}'.format(self.base_counts))
        
    def get_data(self):
        data = {'t': self.t, 'regret': self.total_regret}
        return data

class SmoothedAlgorithm(object):
    def __str__(self):
        return 'Smoothed base algorithm'
        # Algorithm 3 in https://proceedings.neurips.cc/paper/2020/hash/751d51528afe5e6f7fe95dece4ed32ba-Abstract.html
        # Please contact the authors for implementation details 

class SmoothCorral:
    def __str__(self):
        return 'Smooth Corral'
        # Smooth Corral from https://proceedings.neurips.cc/paper/2020/hash/751d51528afe5e6f7fe95dece4ed32ba-Abstract.html
        # Please contact the authors for implementation details

class Corral_base:
    def __str__(self):
        return 'Base Corral'
        # Corral Base invoked by LinUCBPlusCorral
        # Adapted from Smooth Corral

class LinUCBPlusCorral(object):
    def __str__(self):
        return 'LinUCB++ with Carrol (ours)'
        # Please implement Corral_base first
    def __init__(self, the_problem, beta, expressive, double_steps=True):
        self.beta = beta
        self.expressive = expressive
        self.double_steps = double_steps
        self.the_problem = the_problem
        self.d = the_problem['d']
        self.theta_star = the_problem['theta_star']
        self.T = the_problem['T']
        self.stddev = the_problem['stddev']
        self.lambda_para = the_problem['lambda_para']
        if self.expressive == True:
            self.action_set = the_problem['action_set_expressive']
            self.true_reward = the_problem['true_reward_expressive']
            self.opt_reward = the_problem['opt_reward_expressive']
            self.min_reward = the_problem['min_reward_expressive']
            self.range_reward = self.opt_reward - self.min_reward
        else:
            self.action_set = the_problem['action_set_origin']
            self.true_reward = the_problem['true_reward_origin']
            self.opt_reward = the_problem['opt_reward_origin']
            self.min_reward = the_problem['min_reward_origin']
            self.range_reward = self.opt_reward - self.min_reward
        
        self.K = self.action_set.shape[0]
        self.p = int(np.ceil(np.log2(self.T**beta)))
        self.empirical_frequency = np.zeros((self.p, self.K+self.p-1))
        # row i (start from 0) represents the empirical sampling frequency at iteration (i+1);
        # the additional self.p - 1 columns are for virtual arms.
        self.true_reward_extra = np.zeros(self.p-1)
        # p-1 dimensional matrix that stores true reward of virtual actions
        self.t = 0
        self.regret = 0
        self.i = 1
        self.di = max(min(2**(self.p + 2 - self.i), self.d),1)
        self.Delta_Ti = min(2**(self.p + self.i), self.T)
        self.base = LinUCB_base(self.the_problem, self.di, self.Delta_Ti, expressive=False)
        
    def update(self):
        if self.t == (2**self.p) * (2**(self.i+1)-2)  and self.t!= 0:
            # time when we need to prepare for the next iteration
            self.i += 1
            self.di = max(min(2**(self.p + 2 - self.i), self.d),1)
            self.Delta_Ti = min(2**(self.p + self.i), self.T)
            self.Ki = self.K + self.i - 1
            # Ki represents the total number of arms, including all virtual arms
            if self.i > 1:
                if self.expressive == True:
                    self.true_reward_extra[self.i-1-1] = \
                        np.inner(self.empirical_frequency[self.i-2][:self.K], self.true_reward) \
                        + np.inner(self.empirical_frequency[self.i-2][self.K:], self.true_reward_extra)
                else:
                    self.true_reward_extra[self.i-1-1] = \
                        np.inner(self.empirical_frequency[self.i-2][:self.K], self.true_reward)+ \
                        np.inner(self.empirical_frequency[self.i-2][self.K:], self.true_reward_extra)
                # we are currently in iteration i, prepare for virtual arm in iteration i-1
                # we have another -1 here since index starts from 0
            linucb_base = LinUCB_base(self.the_problem, self.di, self.Delta_Ti, self.expressive)
            virtual_arm = self.empirical_frequency[:self.i-1,:]
            virtual_true_reward = self.true_reward_extra
            ucb_base = UCB_base(self.the_problem, virtual_arm, virtual_true_reward, self.Delta_Ti, \
                self.p, self.expressive)
            bases = [SmoothedAlgorithm(linucb_base, self.double_steps), SmoothedAlgorithm(ucb_base, self.double_steps)]
            self.base = Corral_base(self.the_problem, bases, self.Delta_Ti, self.expressive, \
                self.double_steps)

        if self.i == 1:
            reward, instant_regret, arm = self.base.update()
            self.empirical_frequency[self.i-1, arm] += 1/self.Delta_Ti
            self.regret += instant_regret
            self.t += 1
        elif self.double_steps == False:
            reward, instant_regret, arm_set = self.base.update()
            self.empirical_frequency[self.i-1, arm_set[0]] += 1/self.Delta_Ti
            self.regret += instant_regret
            self.t += 1
        else: 
            reward, instant_regret, arm_set = self.base.update()
            self.empirical_frequency[self.i-1, arm_set[0]] += 1/self.Delta_Ti
            self.empirical_frequency[self.i-1, arm_set[1]] += 1/self.Delta_Ti
            self.regret += instant_regret
            self.t += 2

        if self.t % 500 == 0:
            print('LinUCB++ with Corral, time = {} iteration = {}'.format(self.t, self.i))
            print('frequency of pulling virtual arms = ', self.empirical_frequency[:, -self.p+1:])

    def get_data(self):
        data = {'t': self.t, 'regret': self.regret}
        return data