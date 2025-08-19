# please running following code by providing export data from injest.sql query  
# fllowing code just return top 500 cases .... please do not change capacity and lambda weight .. 
# for more understanding please see the power point files attached in repo. :)

import pandas as pd
from outreach_paused import build_outreach
df = pd.read_csv("data25.csv", parse_dates=["week"])
active_q, paused_q, model = build_outreach(df, lambda_blend=0.6, capacity=500)
