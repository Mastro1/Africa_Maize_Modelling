"""This script is used to delete the files that tried to extract admin2 where it was not possible"""

import os
import pandas as pd

remote_sensing_dir = "ASR/GADM/remote_sensing"
list_of_files = os.listdir(remote_sensing_dir)

for file in list_of_files:
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(remote_sensing_dir, file))
        if df["PCODE"].isna().all():
            os.remove(os.path.join(remote_sensing_dir, file))
            print(f"Deleted {file} because it has no PCODE")



