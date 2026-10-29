import pandas as pd

rawdata = pd.read_csv('StudyResult.csv')
latinSquare = pd.read_csv('latin_square_5conditions.csv', header=None)
newdata = rawdata.copy(deep=True)

#print(rawdata)
#print(latinSquare)

for (index, row) in rawdata.iterrows():
    seq = latinSquare.iloc[index]
    for i in range(len(seq)):
        if seq[i] == 'A':
            new_id = 1
        elif seq[i] == 'B':
            new_id = 2
        elif seq[i] == 'C':
            new_id = 3
        elif seq[i] == 'D':
            new_id = 4
        elif seq[i] == 'E':
            new_id = 5

        for j in range(1,6):
            newdata.loc[index, 'I%d-%d'%(new_id, j)] = rawdata.iloc[index]['I%d-%d'%(i+1, j)]


print(newdata)

newdata.columns=['filename', 'DCSM-1', 'DCSM-2', 'DCSM-3', 'DCSM-4', 'DCSM-5', 'HumanBaseline-1', 'HumanBaseline-2', 'HumanBaseline-3',
 'HumanBaseline-4', 'HumanBaseline-5', 'Ours-1', 'Ours-2', 'Ours-3', 'Ours-4', 'Ours-5',
 'PathGAN-1', 'PathGAN-2', 'PathGAN-3', 'PathGAN-4', 'PathGAN-5', 'Saltinet-1', 'Saltinet-2', 'Saltinet-3', 'Saltinet-4', 'Saltinet-5']

print(newdata.sum(axis=0))

with open('ResultFinal.csv','w') as write_csv:
    write_csv.write(newdata.to_csv(index=False))
