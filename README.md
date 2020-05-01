# Containing-the-Spread-of-Coronavirus-Disease-2019-COVID-19-Meteorological-Factors-and-Interventio
Repository includes collected case data of various provinces in China and foreign data. At the same time, we give the code of the SEIR model, which is used to fit the data and find important parameter values.
In the repository, we collected the individual level datasets of 21 provinces (Anhui, Chongqing, Gansu, Guangzhou, Guangxi, Hainan, Hebei, Heilongjiang, Henan, Inner Mongolia, Jiangsu, Jilin, Liaoning, Ningxia, Shaanxi, Sichuan, Tianjin, Yunnanm, Zhejiang) in China mainland and Hongkong, China. 
Each dataset contains some key information, including patient case, city, gender, age, whether imported patient, departure date, return date, infected date, onset date, date of seeking care at a hospital or clinic, isolated date, confirmed date.
11 provinces of these have all information, and others have part of information.
From the dataset, we can extract the data to be fitted, and obtain the impact of initial patients. And we can calculate incubation period, the average treatment time, the average isolation time.
Using these data, the SEIR model can fit the real data and find the other important parameters values, such as transmission rate and pre-symptomatic transmission period.
