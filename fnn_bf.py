# Main Variable
ESS_RATIO = 50
RE = "Wind"
#RE = "Solar"
address = "Belgium_Data/"

# Main Parameters
learning_rate = 1e-5
total_epoch = 10000
print_interval = 500

import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, learning_rate, in_size, hidden_size, out_size):
        super(NN, self).__init__()
        self.learning_rate = learning_rate
        self.in_size       = in_size
        self.hidden_size   = hidden_size
        self.out_size      = out_size
        
        self.layer1 = nn.Linear(self.in_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.out_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = self.fc_out(x)
        return out
        
    def train_net(self, x, y):
        x, y = torch.tensor(x,dtype=torch.float), torch.tensor(y,dtype=torch.float)
        loss = F.mse_loss(self.forward(x), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

in_size       = past
hidden_size   = 16
out_size      = 1

model = NN(learning_rate, in_size, hidden_size, out_size)
batch_size = 128

train_input = np.zeros((size_train-past, past))
train_output = np.zeros((size_train-past, 1))
for i in range(size_train-past):
    train_input[i,:] = np.reshape(data_train[i:i+past], (4))
    train_output[i,:] = data_train[i+past][0]

val_input = np.zeros((size_val-past, past))
val_output = np.zeros((size_val-past, 1))
for i in range(size_val-past):
    val_input[i,:] = np.reshape(data_val[i:i+past], (4))
    val_output[i,:] = data_val[i+past][0]

test_input = np.zeros((size_test-past, past))
test_output = np.zeros((size_test-past, 1))
for i in range(size_test-past):
    test_input[i,:] = np.reshape(data_test[i:i+past], (4))
    test_output[i,:] = data_test[i+past][0]

total_batch = int((size_train-past)/batch_size) + 1
printing = -1
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

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_x = train_input[batch_size*i:batch_size*(i+1),:]
        batch_y = train_output[batch_size*i:batch_size*(i+1),:]
        model.train_net(batch_x, batch_y)

    if epoch == 0 or (epoch+1) % print_interval == 0:
        printing += 1
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

        train_predict = model.forward(torch.tensor(train_input, dtype=torch.float)).detach().numpy()
        state = [data_train[j][0] for j in range(past)] + [E_max/2]
        for i in range(len(train_predict)):
            action = train_predict[i][0]
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
            state = next_state[:]
            
            pred_train[printing] += [action]
            mape_train[printing] += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
            score_train[printing] += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
            mean_cost_train[printing] += [-reward]
        
        val_predict = model.forward(torch.tensor(val_input, dtype=torch.float)).detach().numpy()
        state = [data_val[j][0] for j in range(past)] + [E_max/2]
        for k in range(len(val_predict)):
            action = val_predict[k][0]
            real = data_val[k+past][0]
            E_s = state[past]
            P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - E_s)/tdelta)
            P_dlimit = min(P_dmax, eff_d*(E_s - E_max*soc_min)/tdelta)
            P_c = min(max(real-action, 0), P_climit)
            P_d = min(max(action-real, 0), P_dlimit)
            E_s_prime = E_s + eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
            reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                            - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
            next_state = [data_val[k+1+j][0] for j in range(past)] + [E_s_prime]
            state = next_state[:]
            
            pred_val[printing] += [action]
            mape_val[printing] += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
            score_val[printing] += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
            mean_cost_val[printing] += [-reward]
        
        test_predict = model.forward(torch.tensor(test_input, dtype=torch.float)).detach().numpy()
        state = [data_test[j][0] for j in range(past)] + [E_max/2]
        for l in range(len(test_predict)):
            action = test_predict[l][0]
            real = data_test[l+past][0]
            E_s = state[past]
            P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - E_s)/tdelta)
            P_dlimit = min(P_dmax, eff_d*(E_s - E_max*soc_min)/tdelta)
            P_c = min(max(real-action, 0), P_climit)
            P_d = min(max(action-real, 0), P_dlimit)
            E_s_prime = E_s + eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
            reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                            - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
            next_state = [data_test[l+1+j][0] for j in range(past)] + [E_s_prime]
            state = next_state[:]
            
            pred_test[printing] += [action]
            mape_test[printing] += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
            score_test[printing] += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
            mean_cost_test[printing] += [-reward]

        MAPE_train = round(100*np.mean(mape_train[printing]),2)
        MAPE_val = round(100*np.mean(mape_val[printing]),2)
        MAPE_test = round(100*np.mean(mape_test[printing]),2)
        Score_train = round(np.mean(score_train[printing]),3)
        Score_val = round(np.mean(score_val[printing]),3)
        Score_test = round(np.mean(score_test[printing]),3)
        Mean_Cost_train = int(RE_Capacity*np.mean(mean_cost_train[printing]))
        Mean_Cost_val = int(RE_Capacity*np.mean(mean_cost_val[printing]))
        Mean_Cost_test = int(RE_Capacity*np.mean(mean_cost_test[printing]))

        print("epoch: {}".format(epoch+1))
        print("MAPE_train: {}%, MAPE_val: {}%, MAPE_test: {}%".format(MAPE_train, MAPE_val, MAPE_test))
        print("Score_train: {}, Score_val: {}, Score_test: {}".format(Score_train, Score_val, Score_test))
        print("Mean_Cost_train: ${}, Mean_Cost_val: ${}, Mean_Cost_test: ${}".format(Mean_Cost_train, Mean_Cost_val, Mean_Cost_test))
        print("----------------------------------------------------------")

# Produce outputs from the validation set
#
#select_num = np.argmin(np.mean(mape_val,axis=1))
#select = pd.DataFrame(np.array(pred_test[select_num][:]))
#select.to_csv(address+RE+"_BF.csv")