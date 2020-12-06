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
    'measles_Spreader_atRisk'
]

numeric_features = [
    'AvgHouseholdExpenseOnPresents',
    'AvgHouseholdExpenseOnSocialGames',
    'AvgHouseholdExpenseParkingTicketsPerYear',
    'AvgMinSportsPerDay',
    'AvgTimeOnSocialMedia',
    'AvgTimeOnStuding',
    'BMI',
    'DateOfPCRTest',
    'NrCousins',
    'StepsPerYear',
    'TimeOnSocialActivities',
    'pcrResult1',
    'pcrResult2',
    'pcrResult3',
    'pcrResult4',
    'pcrResult5',
    'pcrResult6',
    'pcrResult7',
    'pcrResult8',
    'pcrResult9',
    'pcrResult10',
    'pcrResult11',
    'pcrResult12',
    'pcrResult13',
    'pcrResult14',
    'pcrResult15',
    'pcrResult16',
    'CurrentLocation_Lat',
    'CurrentLocation_Long']


categorical_features = [
    'AgeGroup',
    'DisciplineScore',
    'HappinessScore',
    'Sex'
]


continous_features = ['StepsPerYear',
                    'TimeOnSocialActivities',
                    'AvgHouseholdExpenseOnPresents',
                    'AvgHouseholdExpenseOnSocialGames',
                    'AvgHouseholdExpenseParkingTicketsPerYear',
                    'AvgMinSportsPerDay',
                    'AvgTimeOnSocialMedia',
                    'AvgTimeOnStuding',
                    'BMI',
                    'pcrResult1',
                    'pcrResult10',
                    'pcrResult11',
                    'pcrResult12',
                    'pcrResult13',
                    'pcrResult14',
                    'pcrResult15',
                    'pcrResult16',
                    'pcrResult2',
                    'pcrResult3',
                    'pcrResult4',
                    'pcrResult5',
                    'pcrResult6',
                    'pcrResult7',
                    'pcrResult8',
                    'pcrResult9'
]


positive_scaled_features = ['AgeGroup',
    'AvgHouseholdExpenseOnPresents',
    'AvgHouseholdExpenseOnSocialGames',
    'AvgHouseholdExpenseParkingTicketsPerYear',
    'AvgMinSportsPerDay',
    'AvgTimeOnSocialMedia',
    'AvgTimeOnStuding',
    'BMI',
    'DateOfPCRTest',
    'DisciplineScore',
    'HappinessScore',
    'NrCousins',
    'StepsPerYear',
    'TimeOnSocialActivities',
    'pcrResult14',
    'pcrResult16']

negative_scaled_features = [
    'pcrResult1',
    'pcrResult10',
    'pcrResult11',
    'pcrResult12',
    'pcrResult13',
    'pcrResult15',
    'pcrResult2',
    'pcrResult3',
    'pcrResult4',
    'pcrResult5',
    'pcrResult6',
    'pcrResult7',
    'pcrResult8',
    'pcrResult9'
]

bad_features=['Address', 'Job', 'PatientID']

pre_final_list = ['AvgMinSportsPerDay', 'AvgTimeOnSocialMedia', 'AvgHouseholdExpenseParkingTicketsPerYear',
                  'AvgMinSportsPerDay', 'AvgTimeOnSocialMedia', 'BMI', 'pcrResult1', 'pcrResult12', 'pcrResult14',
                  'pcrResult16', 'pcrResult2', 'pcrResult4', 'pcrResult9', 'CurrentLocation_Long',
                  'AvgHouseholdExpenseOnPresents', 'HappinessScore', 'pcrResult10', 'pcrResult3', 'pcrResult5']

pcrs = list(set(['pcrResult1', 'pcrResult12', 'pcrResult14', 'pcrResult16', 'pcrResult2', 'pcrResult4', 'pcrResult9',
                 'pcrResult1', 'pcrResult10', 'pcrResult12', 'pcrResult13',
                'pcrResult14', 'pcrResult16', 'pcrResult2', 'pcrResult3', 'pcrResult4', 'pcrResult5', 'pcrResult9']))
others = list(set(['AgeGroup', 'AvgHouseholdExpenseOnSocialGames', 'AvgHouseholdExpenseParkingTicketsPerYear',
                   'AvgMinSportsPerDay', 'AvgTimeOnSocialMedia', 'BMI', 'AvgHouseholdExpenseOnPresents',
                   'AvgHouseholdExpenseOnSocialGames','HappinessScore', 'NrCousins', 'StepsPerYear',
                   'TimeOnSocialActivities']))