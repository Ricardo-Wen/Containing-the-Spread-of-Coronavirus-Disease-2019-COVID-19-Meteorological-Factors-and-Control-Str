from __future__ import division
import numpy as np
import math
import scipy.integrate as spi

#fitting data
#data_ti = [5, 11, 17, 21, 29, 34, 39, 42, 46, 48, 49, 53, 54, 56, 58, 60]#Lingning Province
#data_ti = [3, 9,  11, 19, 22, 30, 36, 41, 43, 49, 52, 56, 57, 60, 61, 61]#Jilin Province
#data_ti = [6, 12, 17, 23, 25, 33, 43, 56, 62, 67, 71, 73, 77, 77, 77, 81, 84]#Tianjin
#data_ti = [13, 32, 45, 58, 63, 79, 86, 99,108, 117, 119, 125, 128, 129, 131, 131]#Shaanxi Province
#data_ti = [31, 73, 86, 101, 138, 169, 187, 209, 232, 245, 258, 276, 288, 305, 319, 332, 350]#Chongqing
#data_ti = [59, 120, 188, 243, 287, 359, 388, 428, 460, 503, 522, 553, 591, 613, 629, 654, 665]#Anhui Province
#data_ti = [63, 120, 171, 219, 249, 306, 366, 411, 443, 479, 507, 536, 561, 578, 581, 588]#Henan Province
#data_ti = [23, 53, 93, 120, 160, 192, 232, 259, 288, 320, 327, 361, 376, 379, 380, 383, 389]#Shandong Province
#data_ti = [10, 13, 17, 21, 22, 23, 27, 37, 39, 42, 42, 43, 46]#Yunnan Province
#data_ti = [11, 26, 39, 55, 63, 70, 79, 90, 93, 98, 110, 111, 119, 128, 134, 137, 143]#Guangxi Provence
data_ti = [20, 42, 80, 89, 105, 121, 160, 186, 212, 241, 268, 297, 305, 314, 329, 337]#Helongjiang Province

#data_ti = [0, 0, 1, 4, 4, 4, 7, 7, 11, 13, 15, 24, 28, 36, 39, 44, 46, 48, 58, 67, 77, 89, 101]#Wuhan City

#data_ti = [2, 4, 8, 12, 13, 14, 22, 23, 29, 29, 32, 33, 38, 41, 42]#Hongkong period 1
#data_ti = [5, 6, 7, 15, 16, 23, 23, 26, 27, 33, 36, 37, 37, 38, 40]#Hongkong period 2
#data_ti = [8, 9, 16, 16, 19, 20, 26, 29, 30, 30, 31, 33, 35, 36, 38]#Hongkong period 3
#data_ti = [0, 3, 4, 10, 13, 14, 14, 15, 17, 19, 21, 23, 23, 23, 23]#Hongkong period 4
#data_ti = [6, 9, 10, 10, 11, 13, 15, 17, 19, 19, 19, 19, 19, 22, 22]#Hongkong period 5
#data_ti = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 12, 12, 13, 15, 16]#Hongkong period 6
#data_ti = [2, 4, 6, 6, 6, 6, 6, 9, 9, 10, 12, 13, 13, 14, 14]#Hongkong period 7
#data_ti = [0, 0, 0, 0, 3, 3, 4, 6, 7, 7, 8, 8, 12, 12, 14]#Hongkong period 8

#Initial case
#data_wy = [14, 13, 9, 5, 4, 3, 3, 1, 1, 1, 1]#Lingning Province
#data_wy = [11, 10, 6, 4, 3, 2, 2, 1]#Jilin Province
#data_wy = [5, 4, 4, 4, 1, 1, 1, 1, 1, 1]#Tianjin
#data_wy = [25, 21, 14, 12, 11, 6, 4, 3, 2, 2, 2, 1, 1, 1]#Shaanxi Province
#data_wy = [81, 69, 57, 47, 44, 33, 23, 17, 14, 9, 2, 1, 1, 1, 1, 1]#Chongqing
#data_wy = [125, 100, 81, 61, 46, 33, 26, 20, 16, 10, 7, 2, 1]#Anhui Province
#data_wy = [141, 104, 80, 65, 51, 40, 30, 22, 15, 12, 11, 9, 6, 5, 5, 3]#Henan Province
#data_wy = [32, 25, 18, 15, 12, 10, 7, 5, 5, 3, 3, 2, 1, 1]#Shandong Province
#data_wy = [26, 25, 16, 6, 5, 4, 3, 3, 3, 1]#Yunnan Province
#data_wy = [21, 20, 17, 15, 12, 10, 9, 5, 2, 2, 2, 2, 2]#Guangxi Provence
data_wy = [37, 29, 19, 16, 15, 11, 9, 7, 3, 3, 3, 3, 3, 2, 2, 2]#Helongjiang Province

#data_wy = []#Wuhan City

#data_wy = [14, 14, 14, 12, 11, 9, 9, 9, 7, 5, 2, 2, 1, 1, 1]#Hongkong period 1
#data_wy = [20, 18, 16, 16, 16, 14, 12, 5, 5, 2, 2, 1, 1, 1, 1]#Hongkong period 2
#data_wy = [21, 21, 19, 16, 6, 6, 3, 2, 1, 1, 1, 1]#Hongkong period 3
#data_wy = [23, 13, 13, 9, 8, 7, 6, 3, 3, 2, 1, 1, 1]#Hongkong period 4
#data_wy = [13, 11, 10, 9, 6, 5, 4, 2, 2, 2]#Hongkong period 5
#data_wy = [19, 15, 14, 10, 8, 8, 7, 4, 4, 4, 2, 1, 1, 1]#Hongkong period 6
#data_wy = [13, 11, 11, 10, 7, 7, 6, 2, 1, 1, 1]#Hongkong period 7
#data_wy = [16, 11, 9, 6, 2, 1, 1, 1]#Hongkong period 8

#population
#N = 43590000#Lingning Province
#N = 27040000#Jilin Province
#N = 15600000#Tianjin
#N = 38640000#Shaanxi Province
#N = 35623100#Chongqing
#N = 63236000#Anhui Province
#N = 96050000#Henan Province
#N = 100000000#Shandong Province
#N = 48300000#Yunnan Province
#N = 49260000#Guangxi Provence
N = 37731000#Helongjiang Province

#N = 9083500#Wuhan City

#N = 7451000#Hongkong

#isolation rate
#alpha = 17/60#Lingning Province
#alpha = 0.3651#Jilin Province
#alpha = 10/94#Tianjin
#alpha = 35/133#Shaanxi Province
#alpha = 0.22145#Chongqing
#alpha = 0.09789#Anhui Province
#alpha = 0.02385#Henan Province
#alpha = 0.015424#Shandong Province
#alpha = 0.087#Yunnan Province
#alpha = 8/143#Guangxi Provence
alpha = 0.07418#Helongjiang Province

#alpha = 0#Wuhan City

#alpha = 0#Hongkong period 1
#alpha = 0#Hongkong period 2
#alpha = 0#Hongkong period 3
#alpha = 0#Hongkong period 4
#alpha = 0.0455#Hongkong period 5
#alpha = 0.125#Hongkong period 6
#alpha = 0.1429#Hongkong period 7
#alpha = 0.2143#Hongkong period 8


#the average treatment time
#l = 1.1628#Lingning Province
#l = 1.65#Jilin Province
#l = 3.023#Tianjin
#l = 2.58#Shaanxi Province
#l = 2.5778#Chongqing
#l = 2.7129#Anhui Province
#l = 2.8062#Henan Province
#l = 2.2428#Shandong Province
#l = 2.381#Yunnan Province
#l = 2.9259#Guangxi Provence
l = 3.0994#Helongjiang Province

#l = 12.5#Wuhan City

#l = 7.1905#Hongkong period 1
#l = 6.775#Hongkong period 2
#l = 6.6316#Hongkong period 3
#l = 7.3478#Hongkong period 4
#l = 6.7143#Hongkong period 5
#l = 3.5#Hongkong period 6
#l = 2.0833#Hongkong period 7
#l = 0.7273#Hongkong period 8

#SEIR model
def func3SEIRS(inivalue,t):#两阶段
    Y = np.zeros(9)
    X = inivalue

    if (t < len(data_wy)):
        i = int(t)
        Y[0] = - ((beta * X[0] * (X[2] + X[3] + X[5] + data_wy[i])) / N)
        # 潜伏个体变化
        Y[1] = ((beta * X[0] * (X[2] + X[3] + X[5]+ data_wy[i])) / N) - X[1] / x
        Y[2] = (1-alpha) * X[1] / x - X[2] / TH
        Y[3] = X[2] / TH - X[3] / l
        Y[4] = X[3] / l
        Y[5] = alpha * X[1] / x - X[5] / (TH-isolotion_time)
        Y[6] = X[5] / (TH-isolotion_time) - X[6] / isolotion_time
        Y[7] = X[6] / isolotion_time
        Y[8] = X[2] / TH
    else:
        Y[0] = - ((beta * X[0] * (X[2] + X[3] + X[5])) / N)
        # 潜伏个体变化
        Y[1] = ((beta * X[0] * (X[2] + X[3] + X[5])) / N) - X[1] / x
        Y[2] = (1-alpha) * X[1] / x - X[2] / TH
        Y[3] = X[2] / TH - X[3] / l
        Y[4] = X[3] / l
        Y[5] = alpha * X[1] / x - X[5] / (TH-isolotion_time)
        Y[6] = X[5] / (TH-isolotion_time) - X[6] / isolotion_time
        Y[7] = X[6] / isolotion_time
        Y[8] = X[2] / TH

    return Y

#update SEIR model
def func3newSEIRS(inivalue,t):#两阶段
    Y = np.zeros(9)
    X = inivalue

    if (t < len(data_wy)):
        i = int(t)
        Y[0] = - ((betanew * X[0] * (X[2] + X[3] + X[5] + data_wy[i])) / N)
        # 潜伏个体变化
        Y[1] = ((betanew * X[0] * (X[2] + X[3] + X[5] + data_wy[i])) / N) - X[1] / x
        Y[2] = (1-alpha) * X[1] / x - X[2] / TH
        Y[3] = X[2] / TH - X[3] / l
        Y[4] = X[3] / l
        Y[5] = alpha * X[1] / x - X[5] / (TH-isolotion_time)
        Y[6] = X[5] / (TH-isolotion_time) - X[6] / isolotion_time
        Y[7] = X[6] / isolotion_time
        Y[8] = X[2] / TH
    else:
        Y[0] = - ((betanew * X[0] * (X[2] + X[3] + X[5])) / N)
        # 潜伏个体变化
        Y[1] = ((betanew * X[0] * (X[2] + X[3] + X[5])) / N) - X[1] / x
        Y[2] = (1-alpha) * X[1] / x - X[2] / TH
        Y[3] = X[2] / TH - X[3] / l
        Y[4] = X[3] / l
        Y[5] = alpha * X[1] / x - X[5] / (TH - isolotion_time)
        Y[6] = X[5] / (TH - isolotion_time) - X[6] / isolotion_time
        Y[7] = X[6] / isolotion_time
        Y[8] = X[2] / TH

    return Y

#this function is used to calculate the initial infectious cases
def SUM_OF_I1_0(x):
    if x >= 1:
        x1 = int(x)-1
        x2 = x- int(x)
        y = (1-x2)*data_ti[x1]+x2*data_ti[x1+1]

    elif x<1:
        y = x * data_ti[0]
    return y

#this function is used to calcuated the overall error using the least squares
def error(RES):
    y = 0
    for i in range(1,len(data_ti)+1):
        y=y+((RES[i, 8] + RES[i, 7])-(data_ti[i-1]))**2
    return y

#Incubation period
#Te = 6.24
#Te = 5.8
Te = 5.5

#Fitting time period
T = len(data_ti)
T_range = np.arange(0, T+31)

#x is the no-infectious period in incubation period
for x in np.arange(2.72, 2.94, 0.02):

    T = 3000  # initiate temperature
    Tmin = 10  # minimum value of terperature
    k = 100  # times of internal circulation
    t = 0  # time

    #the average isolation time
    #isolotion_time = (5 + 13 * TH) / 17  # Lingning Province
    #isolotion_time = (13 + 16 * TH) / 23  # Jilin Province
    #isolotion_time = (6+5*TH) / 10#Tianjin
    #isolotion_time = (34+14*TH) /35#Shaanxi Province
    #isolotion_time = 0.1719+0.875*TH#Chonqing
    #isolotion_time = 0.5385 + 0.6462 * TH#Anhui Province
    #isolotion_time = (8+7*TH) / 14#Henan Province
    #isolotion_time = (5 + 3 * TH) / 6  # Shandong Province
    # isolotion_time = (1+3*TH)/4#Yunnan Province
    #isolotion_time = (6+4*TH) / 8#Guangxi Province
    isolotion_time = (18+13*TH)/25#Helongjiang Province

    #isolotion_time = 1#Hongkong period 5
    #isolotion_time = 1.5#Hongkong period 6
    #isolotion_time = 1.5#Hongkong period 7
    #isolotion_time = 1.3333#Hongkong period 8

    beta = 0.03448  # initial transmission rate
    TH = Te - x  # pre-symptomatic transmission period

    #initial state
    R1_0 = R2_0 = R3_0 = 0
    I_0 = I3_0 = 0
    I1_0 = (1-alpha) * SUM_OF_I1_0(TH)
    I2_0 = alpha * SUM_OF_I1_0(TH)
    E_0 = 50
    S_0 = N - E_0 - I1_0 - I2_0

    INI = (S_0, E_0, I1_0, I3_0, R3_0, I2_0, R1_0, R2_0, I_0)
    RES = spi.odeint(func3SEIRS, INI, T_range)

    ymin = error(RES)
    betamin = 0.03448
    E_0min = 50
    RESmin = RES

    #Simulated annealing process, iteratively find the optimal parameter value:beta E_0
    while T >= Tmin:
        for i in range(k):
            INI = (S_0, E_0, I1_0, I3_0, R3_0, I2_0, R1_0, R2_0, I_0)
            RES = spi.odeint(func3SEIRS, INI, T_range)
            y = error(RES)
            betanew = beta + np.random.uniform(low=-0.1, high=0.1)
            E_0new = E_0 + np.random.uniform(low=-10, high=10)

            if (0 <= betanew and betanew <= 1 and E_0new > 0):
                S_0new = N - E_0new - I1_0 - I2_0
                INInew = (S_0new, E_0new, I1_0, I3_0, R3_0, I2_0, R1_0, R2_0, I_0)
                RESNew = spi.odeint(func3newSEIRS, INInew, T_range)
                yNew = error(RESNew)

                # print(yNew, y)

                if yNew - y < 0:
                    beta = betanew
                    E_0 = E_0new
                    S_0 = S_0new
                    y = yNew
                    RES = RESNew

                    if (y < ymin):
                        betamin = beta
                        E_0min = E_0
                        ymin = y
                        RESmin = RES
                        #print(ymin, RESmin)

                else:
                    p = math.exp(-(yNew - y) / T)
                    r = np.random.uniform(low=0, high=1)

                    if r < p:
                        beta = betanew
                        E_0 = E_0new
                        S_0 = S_0new
                        y = yNew
                        RES = RESNew
        t += 1

        T = 3000 / (1 + t)

    print(x, ymin, betamin, E_0, SUM_OF_I1_0(TH), (1-alpha)*betamin*(l+TH)+alpha*betamin*(TH-isolotion_time))