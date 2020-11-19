# Main Variable
ESS_RATIO_X = [25, 50]
RE_X = ["Solar", "Wind"]
address = "results/"

import pandas as pd
import numpy as np

for RE in RE_X:
  for ESS_RATIO in ESS_RATIO_X:

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

    real_csv = pd.read_csv(address+RE+'_Belgium_19.csv', index_col=0)
    BF_csv = pd.read_csv(address+RE+'_BF_Select.csv', index_col=0)
    ECF_csv = pd.read_csv(address+RE+'_ECF_'+str(ESS_RATIO)+'_Select.csv', index_col=0)

    real0 = np.zeros((int(len(real_csv)/unit),1))
    BF0 = np.zeros((int(len(BF_csv)/unit),1))
    ECF0 = np.zeros((int(len(ECF_csv)/unit),1))

    REAL = []
    for i in range(len(real0)):
        j = i*unit
        real0[i] = round(pd.Series.mean(real_csv['Power(MW)'][j:j+unit])/RE_Capacity, 3)
        if real0[i] != 0:
            REAL += [real0[i]]
    REAL = np.reshape(np.array(REAL[past:]), (len(REAL[past:]),1))
    BF = np.array(BF_csv)
    ECF = np.array(ECF_csv)

    mape_bf = []
    mape_ecf = []
    score_bf = []
    score_ecf = []
    mean_cost_bf = []
    mean_cost_ecf = []

    E_s = E_max/2
    for i in range(len(REAL)):
        real = REAL[i][0]
        action = BF[i][0]
        P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - E_s)/tdelta)
        P_dlimit = min(P_dmax, eff_d*(E_s - E_max*soc_min)/tdelta)
        P_c = min(max(real-action, 0), P_climit)
        P_d = min(max(action-real, 0), P_dlimit)
        E_s += eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
        reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                        - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
        
        mape_bf += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
        score_bf += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
        mean_cost_bf += [-reward]

    E_s = E_max/2
    for i in range(len(REAL)):
        real = REAL[i][0]
        action = ECF[i][0]
        P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - E_s)/tdelta)
        P_dlimit = min(P_dmax, eff_d*(E_s - E_max*soc_min)/tdelta)
        P_c = min(max(real-action, 0), P_climit)
        P_d = min(max(action-real, 0), P_dlimit)
        E_s += eff_c*P_c*tdelta - (1/eff_d)*P_d*tdelta
        reward = tdelta*(- b_cost*(P_c + P_d) - l_cost*((1-eff_c)*P_c + (1/eff_d-1)*P_d) \
                        - l_cost*max(real-action-P_c,0) - p_cost*max(action-real-P_d,0))
        
        mape_ecf += [abs(real - P_c + P_d - action)/(real - P_c + P_d)]
        score_ecf += [1 if l_cost*max(real-action-P_c,0) + p_cost*max(action-real-P_d,0) == 0 else 0]
        mean_cost_ecf += [-reward]

    MAPE_BF = round(100*np.mean(mape_bf),2)
    MAPE_ECF = round(100*np.mean(mape_ecf),2)
    Score_BF = round(np.mean(score_bf),3)
    Score_ECF = round(np.mean(score_ecf),3)
    Mean_Cost_BF = int(RE_Capacity*np.mean(mean_cost_bf))
    Mean_Cost_ECF = int(RE_Capacity*np.mean(mean_cost_ecf))

    print("Renewable Source: {}, Emax: {} p.u.".format(RE, ESS_RATIO/100))
    print("MAPE_BF: {}%, MAPE_ECF: {}%".format(MAPE_BF, MAPE_ECF))
    print("Score_BF: {}, Score_ECF: {}".format(Score_BF, Score_ECF))
    print("Mean_Cost_BF: ${}, Mean_Cost_ECF: ${}".format(Mean_Cost_BF, Mean_Cost_ECF))
    print("----------------------------------------------------------")