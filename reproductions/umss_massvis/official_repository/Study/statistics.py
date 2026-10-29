import numpy as np
import pandas as pd
import researchpy as rp

METHODS=['DCSM','HumanBaseline','Ours','PathGAN','Saltinet']

newdata = pd.read_csv('ResultFinal_stats.csv')
newdata.set_index('filename', inplace=True)
res = newdata.loc['SUM']


for method in METHODS:
    distri = []
    print('%s:'%method)
    for i in range(1,6):
        distri.extend([i] * res['%s-%d'%(method,i)])

    distri = np.array(distri)
    print('mean: ', distri.mean(), 'std: ', distri.std())

Saltinet=[]
for i in range(1,6):
    Saltinet.extend([i] * res['Saltinet-%d'%(i)])
Saltinet = pd.DataFrame(Saltinet, dtype=int)

Ours=[]
for i in range(1,6):
    Ours.extend([i] * res['Ours-%d'%(i)])
Ours = pd.DataFrame(Ours, dtype=int)

print(Saltinet, Ours)


print(
rp.ttest(group1= Saltinet, group1_name= "EX",
         group2= Ours, group2_name= "IN"))