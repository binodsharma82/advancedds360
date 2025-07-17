import pandas as pd
import numpy as np
import pickle
import matplotlib as mp


sourcedata = './SourceData/DataScienceSalary_Scrubbed.csv'
df = pd.read_csv(sourcedata)
print(df.head(10))