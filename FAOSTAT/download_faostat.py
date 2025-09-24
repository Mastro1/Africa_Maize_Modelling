import faostat
import os
import pandas as pd

FAOSTAT_DIR = "FAOSTAT"

def main():
    params = faostat.list_pars('QCL') # 'QCL' is the dataset code for crops and livestock
    print(params)

    my_params = {'area': '5100>', 'element': [2312, 2510, 2413], 'item': '56', 'year': list(range(2000, 2024))}
    data = faostat.get_data_df('QCL', pars=my_params, strval=False)
    print(data.head())

    data.to_csv(os.path.join(FAOSTAT_DIR, "faostat_maize.csv"), index=False)

    # The second part creates duplicates for Tanzania and Democratic Republic of the Congo with names that match HarvestStatAfrica and GADM

    df = pd.read_csv(os.path.join(FAOSTAT_DIR, "faostat_maize.csv"))

    # Create copies with different names
    tanzania_copy = df.loc[df['Area'] == "United Republic of Tanzania",].copy()
    tanzania_copy['Area'] = "Tanzania, United Republic of"
    tanzania_gadm_copy = df.loc[df['Area'] == "United Republic of Tanzania",].copy()
    tanzania_gadm_copy['Area'] = "Tanzania"

    drc_copy = df.loc[df['Area'] == "Democratic Republic of the Congo", ].copy()
    drc_copy['Area'] = "DRC"

    # Concatenate original data with the copies
    df = pd.concat([df, tanzania_copy, drc_copy, tanzania_gadm_copy], ignore_index=True)

    df.to_csv(os.path.join(FAOSTAT_DIR, "faostat_maize.csv"), index=False)

    print("Added Tanzania (previous Tanzania, United Republic of) and Democratic Republic of the Congo (previous DRC) duplicates to the dataset")


if __name__ == "__main__":
    main()