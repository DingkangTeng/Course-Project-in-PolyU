# Run in relative path due to VSCode runs in the user directory by defult
import os,sys
os.chdir(sys.path[0])
sys.path.append(os.getcwd())

import pandas as pd

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

def fuzzy_match(row: pd.DataFrame, df2: pd.DataFrame) -> str:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    best_match=process.extractOne(row["Street_NM"], df2["stname"], scorer=fuzz.ratio)
    return best_match


df=pd.read_csv(r"streetShp.csv", dtype=str, nrows=5)
dfSND=pd.read_csv(r"C:\\Users\\tengd\\Desktop\\snd\\win11\\bobaadr.txt", usecols=["sc5", "stname"], dtype=str)

df["fuzzName"]=df.parallel_apply(fuzzy_match, axis=1, df2=dfSND)
df["fuzzName"]=df["fuzzName"].str[0]

df.to_csv(r"streetShpSND.csv", encoding="utf-8")