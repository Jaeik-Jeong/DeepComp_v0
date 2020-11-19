# Main Variable
ESS_RATIO = 50
RE = "Wind"
#RE = "Solar"
address = "Belgium_Data/"

# Main Parameters
learning_rate1   = 1e-4
learning_rate2   = 1e-5
pretrain_episode = 50
total_episode    = 200
print_interval = 10

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO(nn.Module):
    def __init__(self, learning_rate1, learning_rate2, sizes, continuous, out_size_std, gamma, lmbda, eps_clip, K_epoch):
        super(PPO, self).__init__()
        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.sizes          = sizes
        self.continuous     = continuous
        self.gamma          = gamma
        self.lmbda          = lmbda
        self.eps_clip       = eps_clip
        self.K_epoch        = K_epoch
        
        if self.continuous:
            self.out_size = 1
            self.std      = out_size_std
        else:
            self.out_size = out_size_std
        
        self.data = []
        
        self.hidden = nn.ModuleList()
        for k in range(len(sizes)-1):
            self.hidden.append(nn.Linear(sizes[k], sizes[k+1]))
        
        self.fc_pi = nn.Linear(sizes[-1],self.out_size)
        self.fc_v  = nn.Linear(sizes[-1],1)
        self.optimizer1 = torch.optim.Adam(self.parameters(), lr=self.learning_rate1)
        self.optimizer2 = torch.optim.Adam(self.parameters(), lr=self.learning_rate2)

    def pi(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        pi = self.fc_pi(x)
        return pi
    
    def v(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, targ_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done, targ = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            targ_lst.append([targ])
            
        s,a,r,s_prime,done_mask,prob_a,targ = torch.tensor(s_lst, dtype=torch.float), \
                                              torch.tensor(a_lst, dtype=torch.float), \
                                              torch.tensor(r_lst, dtype=torch.float), \
                                              torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), \
                                              torch.tensor(prob_a_lst, dtype=torch.float), \
                                              torch.tensor(targ_lst, dtype=torch.float)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, targ
        
    def pretrain_net(self):
        s, a, r, s_prime, done_mask, prob_a, targ = self.make_batch()
        
        if self.continuous:
            loss = F.mse_loss(self.pi(s), targ)
        else:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(self.pi(s), targ)
        
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()
    
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, targ = self.make_batch()
        
        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            
            if self.continuous:
                prob = torch.distributions.Normal(self.pi(s), self.std)
                prob_new = torch.exp(prob.log_prob(a))
            else:
                prob = F.softmax(self.pi(s), dim=1)
                prob_new = prob.gather(1,a)
            
            ratio = torch.exp(torch.log(prob_new) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.mse_loss(self.v(s), td_target.detach())
            
            self.optimizer2.zero_grad()
            loss.mean().backward()
            self.optimizer2.step()

import pandas as pd
import numpy as np

if RE == "Solar":
    RE_Capacity = 3887
if RE == "Wind":
    RE_Capacity = 3796

past = 4 # The number of past data for input
unit = 4 # Time resolution (unit : 15 minute)
tdelta = unit/4
ESS_SIZE = ESS_RATIO*RE_Capacity/100
MAX_CPOWER = ESS_SIZE/3
MAX_DPOWER = ESS_SIZE/3
E_max = ESS_SIZE/RE_Capacity
P_cmax = MAX_CPOWER/RE_Capacity
P_dmax = MAX_DPOWER/RE_Capacity
b_cost = 10
l_cost = 50
p_cost = 100
eff_c = 0.9
eff_d = 0.9
soc_min = 0.1
soc_max = 0.9

data_train_csv1 = pd.read_csv(address+RE+'_Belgium_16.csv', index_col=0)
data_train_csv2 = pd.read_csv(address+RE+'_Belgium_17.csv', index_col=0)
data_train_csv = pd.concat([data_train_csv1, data_train_csv2])
data_val_csv = pd.read_csv(address+RE+'_Belgium_18.csv', index_col=0)
data_test_csv = pd.read_csv(address+RE+'_Belgium_19.csv', index_col=0)

size_train0 = int(len(data_train_csv)/unit)
data_train0 = np.zeros((size_train0,1))
size_val0 = int(len(data_val_csv)/unit)
data_val0 = np.zeros((size_val0,1))
size_test0 = int(len(data_test_csv)/unit)
data_test0 = np.zeros((size_test0,1))

data_train = []
for i in range(size_train0):
    j = i*unit
    data_train0[i] = round(pd.Series.mean(data_train_csv['Power(MW)'][j:j+unit])/RE_Capacity, 3)
    if data_train0[i] != 0:
        data_train += [data_train0[i]]
data_train = np.reshape(np.array(data_train), (len(data_train),1))

data_val = []
for i in range(size_val0):
    j = i*unit
    data_val0[i] = round(pd.Series.mean(data_val_csv['Power(MW)'][j:j+unit])/RE_Capacity, 3)
    if data_val0[i] != 0:
        data_val += [data_val0[i]]
data_val = np.reshape(np.array(data_val), (len(data_val),1))

data_test = []
for i in range(size_test0):
    j = i*unit
    data_test0[i] = round(pd.Series.mean(data_test_csv['Power(MW)'][j:j+unit])/RE_Capacity, 3)
    if data_test0[i] != 0:
        data_test += [data_test0[i]]
data_test = np.reshape(np.array(data_test), (len(data_test),1))

size_train = len(data_train)
size_val = len(data_val)
size_test = len(data_test)

sizes         = [past+1,16,16]
continuous    = 1

std           = 0.01
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 24

model = PPO(learning_rate1, learning_rate2, sizes, continuous, std, gamma, lmbda, eps_clip, K_epoch)
pred_train = [] # Predicted Value
pred_val = []
pred_test = []
mape_train = [] # Mean Absolute Percentage Error
mape_val = []
mape_test = []
score_train = [] # Completely Compensated Errors
score_val = []
score_test = []
mean_cost_train = [] # Mean Error Compensation Cost
mean_cost_val = []
mean_cost_test = []

for episode in range(total_episode):
    if episode == pretrain_episode:
        print("Reinforcement Learning Start!")
        print("----------------------------------------------------------")
    
    pred_train += [[]]
    pred_val += [[]]
    pred_test += [[]]
    mape_train += [[]]
    mape_val += [[]]
    mape_test += [[]]
    score_train += [[]]
    score_val += [[]]
    score_test += [[]]
    mean_cost_train += [[]]
    mean_cost_val += [[]]
    mean_cost_test += [[]]

    state = [data_train[j][0] for j in range(past)] + [E_max/2]
    i = 0
    while i < size_train-past:
        for t in range(T_horizon):
            pi_out = model.pi(torch.tensor(state, dtype=torch.float))
            action = np.random.normal(float(pi_out), std, 1)[0]
            pdf = torch.distributions.Normal(float(pi_out), std)
            prob = float(torch.exp(pdf.log_prob(action)))
            real = data_train[i+past][0]
            
            E_s = state[past]
            P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - E_s)/tdelta)
            P_dlimit = min(P_dmax, eff_d*(E_s - E_max*soc_min)/tdelta)
            P_c = min(max(real-action, 0), P_climit)
            P_d = min(max(action-real, 0), P_dlimit)
            E_s_prime = E_s + eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
            reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                             - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
            next_state = [data_train[i+1+j][0] for j in range(past)] + [E_s_prime]
            model.put_data((state, action, reward, next_state, prob, False, real))
            state = next_state[:]

            pred_train[episode] += [action]
            mape_train[episode] += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
            score_train[episode] += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
            mean_cost_train[episode] += [-reward]
            i += 1
            if i == size_train-past:
                break
        
        if episode < pretrain_episode:
            model.pretrain_net()
        else:
            model.train_net()
    
    state = [data_val[j][0] for j in range(past)] + [E_max/2]
    for k in range(size_val-past):
        action = abs(float(model.pi(torch.tensor(state, dtype=torch.float))))
        real = data_val[k+past][0]
        
        P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - state[past])/tdelta)
        P_dlimit = min(P_dmax, eff_d*(state[past] - E_max*soc_min)/tdelta)
        P_c = min(max(real-action, 0), P_climit)
        P_d = min(max(action-real, 0), P_dlimit)
        E_s_prime = state[past] + eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
        reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                         - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
        next_state = [data_val[k+1+j][0] for j in range(past)] + [E_s_prime]
        state = next_state[:]
        
        pred_val[episode] += [action]
        mape_val[episode] += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
        score_val[episode] += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
        mean_cost_val[episode] += [-reward]
    
    state = [data_test[j][0] for j in range(past)] + [E_max/2]
    for l in range(size_test-past):
        action = abs(float(model.pi(torch.tensor(state, dtype=torch.float))))
        real = data_test[l+past][0]
        
        P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - state[past])/tdelta)
        P_dlimit = min(P_dmax, eff_d*(state[past] - E_max*soc_min)/tdelta)
        P_c = min(max(real-action, 0), P_climit)
        P_d = min(max(action-real, 0), P_dlimit)
        E_s_prime = state[past] + eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
        reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                         - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
        next_state = [data_test[l+1+j][0] for j in range(past)] + [E_s_prime]
        state = next_state[:]
        
        pred_test[episode] += [action]
        mape_test[episode] += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
        score_test[episode] += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
        mean_cost_test[episode] += [-reward]
    
    if episode == 0 or (episode+1) % print_interval == 0:
        MAPE_train = round(100*np.mean(mape_train[episode]),2)
        MAPE_val = round(100*np.mean(mape_val[episode]),2)
        MAPE_test = round(100*np.mean(mape_test[episode]),2)
        Score_train = round(np.mean(score_train[episode]),3)
        Score_val = round(np.mean(score_val[episode]),3)
        Score_test = round(np.mean(score_test[episode]),3)
        Mean_Cost_train = int(RE_Capacity*np.mean(mean_cost_train[episode]))
        Mean_Cost_val = int(RE_Capacity*np.mean(mean_cost_val[episode]))
        Mean_Cost_test = int(RE_Capacity*np.mean(mean_cost_test[episode]))

        print("episode: {}".format(episode+1))
        print("MAPE_train: {}%, MAPE_val: {}%, MAPE_test: {}%".format(MAPE_train, MAPE_val, MAPE_test))
        print("Score_train: {}, Score_val: {}, Score_test: {}".format(Score_train, Score_val, Score_test))
        print("Mean_Cost_train: ${}, Mean_Cost_val: ${}, Mean_Cost_test: ${}".format(Mean_Cost_train, Mean_Cost_val, Mean_Cost_test))
        print("----------------------------------------------------------")

# Produce outputs from the validation set
#
#select_num = np.argmin(np.mean(mean_cost_val,axis=1))
#select = pd.DataFrame(np.array(pred_test[select_num][:]))
#select.to_csv(address+RE+"_ECF_"+str(ESS_RATIO)+".csv")