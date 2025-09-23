import faostat
import os

FAOSTAT_DIR = "FAOSTAT"

def main():
    params = faostat.list_pars('QCL') # 'QCL' is the dataset code for crops and livestock
    print(params)

    my_params = {'area': '5100>', 'element': [2312, 2510, 2413], 'item': '56', 'year': list(range(2000, 2024))}
    data = faostat.get_data_df('QCL', pars=my_params, strval=False)
    print(data.head())

    data.to_csv(os.path.join(FAOSTAT_DIR, "faostat_maize.csv"), index=False)

if __name__ == "__main__":
    main()