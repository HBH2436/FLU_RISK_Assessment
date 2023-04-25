# made by Henry Bowman Hill
import pandas as pd  
import numpy as np
  
# read_csv function which is used to read the required CSV file
df = pd.read_csv('nhis_raw.csv')
  
# drop function which is used in removing or deleting rows or columns from the CSV files
df = df.drop(columns = ["YEAR", "MORTWT","MORTWTSA","MORTSTAT", "WEIGHT","ANEMIAYR","ASTHMAEV","SERIAL","STRATA","PSU","NHISHID","HHWEIGHT","PERNUM","NHISPID","HHX","FMX","PX","PERWEIGHT","SAMPWEIGHT","FWEIGHT","ASTATFLG","CSTATFLG"])

# Dropping niu/na values
df = df.loc[df["BMICAT"] != 0]
df = df.loc[df["BMICAT"] != 9]
df = df.loc[df["HEALTH"] <= 6]
df = df.loc[df["RACENEW"] <= 505]
df = df.loc[df["VACFLUSH12M"] != 0]
df = df.loc[df["VACFLUSH12M"] <= 3]
df = df.loc[df["FLUPNEUYR"] != 0]
df = df.loc[df["FLUPNEUYR"] <= 3]
df = df.loc[df["REGIONBR"] <= 11]
df['FLUPNEUYR'] = df['FLUPNEUYR'].astype(int)  

# Catagorizing Age 0 for low risk ages 1 for high risk ages
bins = [0,60,np.inf]
labels = [0,1]
df['AGE'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

# dataframe for testing flu infection risk
GFdf = pd.DataFrame.dropna(df)
GFdf = GFdf.drop(columns = ["MORTUCODLD"])
GFdf['FLUPNEUYR'] = GFdf['FLUPNEUYR'] - 1

# random oversampling to compensate for low response rates
max_size = GFdf['FLUPNEUYR'].value_counts().max()
lst = [GFdf]
for FLUPNEUYR_index, group in GFdf.groupby('FLUPNEUYR'):
    lst.append(group.sample((max_size)-len(group), replace=True))
GFdf = pd.concat(lst)

# display 
GFdf.to_csv('nhis_clean.csv')
print(GFdf.head())