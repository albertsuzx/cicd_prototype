# Data Handling
import logging
from logging.handlers import RotatingFileHandler
import pickle
import numpy as np
from pydantic import BaseModel

# Server
import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger as fastapi_logger

# Modeling
import funcs
import sklearn

app = FastAPI()

# Initialize logging
formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler('log/test.log', backupCount=0)
logging.getLogger().setLevel(logging.DEBUG)
fastapi_logger.addHandler(handler)
handler.setFormatter(formatter)
# my_logger = logging.getLogger()
# my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Initialize files
lr = pickle.load(open('data/model.pickle', 'rb'))
features = pickle.load(open('data/list_features.pickle', 'rb'))
# functions to convert raw inputs to group variables
inq_main_to_group = pickle.load(open('data/func_feature2group_inq_main_to_group.pickle', 'rb'))
over_cycle_pay_to_group = pickle.load(open('data/func_feature2group_over_cycle_pay_to_group.pickle', 'rb'))
# dictionary to map group variable to woe
cust_app_woe = pickle.load(open('data/dict_group2woe_cust_app_woe.pickle', 'rb'))
land_woe = pickle.load(open('data/dict_group2woe_land_woe.pickle', 'rb'))
pmt_woe = pickle.load(open('data/dict_group2woe_pmt_woe.pickle', 'rb'))
rem2_past_woe = pickle.load(open('data/dict_group2woe_rem2_past_woe.pickle', 'rb'))
remain_pctg_woe = pickle.load(open('data/dict_group2woe_remain_pctg_woe.pickle', 'rb'))
uc_woe = pickle.load(open('data/dict_group2woe_uc_woe.pickle', 'rb'))
list_woe = [uc_woe, pmt_woe, cust_app_woe, land_woe, remain_pctg_woe, rem2_past_woe]


class Data(BaseModel):
    id: str
    UC_NoInqMain: int
    OverCycle_amt_Min6mth: float
    MostRecentApp_No_Grp: str
    UC_LandVal_Grp: str
    RemainingLoanPct_Grp: str
    Rem2Month_flag_MonthsSince_Grp: str


@app.post("/predict")
def predict(data: Data):

    try:
        # Extract data in correct order
        data_dict = data.dict()
        raw_input = [data_dict[feature] for feature in features]
        app_id = data_dict['id']

        # Convert numeric input to group
        uc_inq_group = inq_main_to_group(raw_input[0])
        over_cycle_pay_group = over_cycle_pay_to_group(raw_input[1])

        group_input = []
        group_input.append(uc_inq_group.tolist())
        group_input.append(over_cycle_pay_group.tolist())
        group_input.extend(raw_input[2:])

        # Apply group to woe conversion
        to_predict = []
        for i, value in enumerate(group_input):
            to_predict.append(list_woe[i][value])
        X = np.array(to_predict)

        # Create and return prediction
        prob = lr.predict_proba(X.reshape(1, -1))

        fastapi_logger.info("Score produced for " + app_id)
        return {"prediction": {'prob': prob[0, 1]}}

    except:
        return {"prediction": "error"}


@app.get("/")
def welcome():
    return {"message":"Welcome to EC-fastapi-pipeline"}


if __name__ == '__main__':

    fastapi_logger.info('****************** Starting Server *****************')
    uvicorn.run(app)
