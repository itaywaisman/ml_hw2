import pandas as pd # data analysis and manipulation tool
import numpy as np # Numerical computing tools
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # another visualization library
import warnings
warnings.filterwarnings('ignore')

file = 'virus_hw2.csv'

label_categories = [
    'not_detected_Spreader_NotatRisk',
    'not_detected_NotSpreader_atRisk',
    'not_detected_NotSpreader_NotatRisk',
    'not_detected_Spreader_atRisk',
    'cold_NotSpreader_NotatRisk',
    'cold_Spreader_NotatRisk',
    'cold_Spreader_atRisk',
    'cold_NotSpreader_atRisk',
    'flue_NotSpreader_NotatRisk',
    'flue_NotSpreader_atRisk',
    'flue_Spreader_NotatRisk',
    'covid_NotSpreader_atRisk',
    'covid_Spreader_NotatRisk',
    'flue_Spreader_atRisk',
    'covid_NotSpreader_NotatRisk',
    'covid_Spreader_atRisk',
    'cmv_NotSpreader_NotatRisk',
    'cmv_Spreader_atRisk',
    'cmv_NotSpreader_atRisk',
    'cmv_Spreader_NotatRisk',
    'measles_Spreader_NotatRisk',
    'measles_NotSpreader_NotatRisk',
    'measles_NotSpreader_atRisk',
    'measles_Spreader_atRisk',
]

parse_dates = ['DateOfPCRTest']
df = pd.read_csv(file, parse_dates=parse_dates)

convert_dict = {
    'Address': str,
    'AgeGroup': pd.CategoricalDtype(categories=range(1,9), ordered=True),
    'BloodType': pd.CategoricalDtype(categories=['AB-', 'A+', 'AB+', 'A-', 'B-', 'O-', 'B+', 'O+']),
    'Job': str,
    'Sex': pd.CategoricalDtype(categories=['F', 'M']),
    'SyndromeClass': pd.CategoricalDtype(categories=range(1,5)),
}

df = df.astype(convert_dict)

long_lat_df = df['CurrentLocation'].str.strip('(Decimal').str.split(', ', expand=True).rename(columns={0:'Lat', 1:'Long'})
df['CurrentLocation_Lat'] = long_lat_df['Lat'].str.strip("')")
df['CurrentLocation_Long'] = long_lat_df['Long'].str.strip("Decimal('").str.rstrip("'))")

convert_dict = {
    'CurrentLocation_Lat': float,
    'CurrentLocation_Long': float,
}

df = df.astype(convert_dict)


splitted_df = df['SelfDeclarationOfIllnessForm'].str.split(';', expand=True)
values = splitted_df.values.flatten()
unique_values = pd.unique(values).tolist()
stripped_unique_values = [str(val).strip(' ') for val in unique_values]

# Split by ; to create a list for each row
df['SelfDeclarationOfIllnessForm_list'] = df['SelfDeclarationOfIllnessForm'].str.split(';')

# Replace NAN values with empty list
isna = df['SelfDeclarationOfIllnessForm_list'].isna()
df.loc[isna, 'SelfDeclarationOfIllnessForm_list'] = pd.Series([[]] * isna.sum()).values

# strip whitespaces
df['SelfDeclarationOfIllnessForm_list'] = [[str(val).strip() for val in list(symptom_list)] for symptom_list in df['SelfDeclarationOfIllnessForm_list'].values]

# Create columns
for column_name in stripped_unique_values:
    df[column_name] = pd.Series([1 if column_name in row else 0 for row in df['SelfDeclarationOfIllnessForm_list']])

# Rename no symptoms column
df.rename(columns={'nan': 'No_Symptoms'}, inplace=True)

# Drop irrelevant features
df.drop(labels=['SelfDeclarationOfIllnessForm','SelfDeclarationOfIllnessForm_list'], axis=1, inplace=True)

f = plt.figure(figsize=(80, 80))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()
a = 0