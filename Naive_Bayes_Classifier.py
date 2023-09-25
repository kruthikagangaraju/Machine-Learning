import pandas as pd
import numpy as np

dat = pd.read_csv('play_tennis.csv')
data.drop(columns=['day'], inplace=True)

# Training
print(data['play'].value_counts())
P_Y = 9/14
P_N = 5/14

print(pd.crosstab(data['outlook'], data['play']))
P_O_N = 0
P_R_N = 2/5
P_S_N = 3/5
P_O_Y = 4/9
P_R_Y = 3/9
P_S_Y = 2/9

print(pd.crosstab(data['temp'], data['play']))
P_Cool_N = 1/5
P_Hot_N = 2/5
P_Mild_N = 2/5
P_Cool_Y = 3/9
P_Hot_Y = 2/9
P_Mild_Y = 4/9

print(pd.crosstab(data['humidity'], data['play']))
P_High_N = 4/5
P_Normal_N = 1/5
P_High_Y = 3/9
P_Normal_Y = 6/9

print(pd.crosstab(data['wind'], data['play']))
P_Strong_N = 3/5
P_Weak_N = 2/5
P_Strong_Y = 3/9
P_Weak_Y = 6/9
