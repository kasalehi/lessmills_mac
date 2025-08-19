import pandas as pd
from ews_model import EWSModel

# Load your subscription weekly data
df = pd.read_csv("sample25.csv", parse_dates=["WeekBeginningDate"], low_memory=False)

# Map columns to expected names
df = df.rename(columns={
    "MembershipID": "member_id",
    "WeekBeginningDate": "week",
    "WeekVisits": "engagement"
})

# Initialize & fit the model
model = EWSModel(k_weeks=6, ema_span=4, drought_churn_weeks=8)
model.fit(df)

# Predict Early Warning Scores
scores = model.predict(df)
print(scores.head())

# Evaluate performance
report = model.evaluate(df)
print(report)
