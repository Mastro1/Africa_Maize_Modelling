import pandas as pd
import os

crop5min = pd.read_csv("ResultsComparison/cropyield5min/aggregate_admin2/maize_yield_admin2_1982_2015.csv")
crop5min["source"] = "cropyield5min"

gdhy = pd.read_csv("ResultsComparison/GDHY/aggregate_admin2/maize_yield_admin2_1981_2016.csv")
gdhy["source"] = "gdhy"

spam = pd.read_csv("ResultsComparison/SPAM/aggregate_admin2/spam_maize_yield_admin2_2000_2020.csv")
spam["source"] = "spam"

concat = pd.concat([crop5min, gdhy, spam])

concat["crop_EN"] = "Maize"

concat.to_csv("ResultsComparison/merged_comparisons.csv", index=False)