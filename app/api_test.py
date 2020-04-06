import requests
to_predict_dict = {'id': 'se0001',
                   'UC_NoInqMain': 1,
                   'OverCycle_amt_Min6mth': -3500,
                   'MostRecentApp_No_Grp': '10 Plus',
                   'UC_LandVal_Grp': 'Missing',
                   'RemainingLoanPct_Grp': '65 to 93',
                   'Rem2Month_flag_MonthsSince_Grp': '3 to 5'}

url = 'http://127.0.0.1:8000/predict'
r = requests.post(url, json=to_predict_dict)
