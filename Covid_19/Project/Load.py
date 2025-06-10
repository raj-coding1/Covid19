import pandas as pd

def LoadData():
    df_recover = pd.read_csv(r'C:\Users\Personal PC\Downloads\covid_19_recovered_v1_lyst1747728719904.csv')
    df_confirm = pd.read_csv(r'C:\Users\Personal PC\Downloads\covid_19_confirmed_v1_lyst1747728690432.csv')
    df_death = pd.read_csv(r'C:\Users\Personal PC\Downloads\covid_19_deaths_v1_lyst1747728711771.csv')