#%%

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import to_long_format
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
n = 1000  # number of individuals
hr_treatment = 1.15  # hazard ratio
lambda0 = 0.1  # baseline hazard rate
max_followup = 1000  # max follow-up time

# Simulate treatment assignment (0 or 1)
treatment = np.random.binomial(1, 0.5, size=n)

# Log hazard ratio
beta = np.log(hr_treatment)

# Linear predictor
linpred = beta * treatment

# Simulate survival times from exponential distribution
# T ~ Exponential(hazard = lambda0 * exp(beta * treatment))
u = np.random.uniform(0, 1, size=n)
survival_time = -np.log(u) / (lambda0 * np.exp(linpred))

# Simulate censoring times
censoring_time = np.random.uniform(0, max_followup, size=n)

# Observed time and event indicator
time = np.minimum(survival_time, censoring_time)
event = survival_time <= censoring_time

# Create DataFrame
df = pd.DataFrame({
    'time': time,
    'event': event.astype(int),
    'treatment': treatment
})

# Simulate different treatment starting times
df.loc[df.treatment == 1, "treatment_start_time"] = np.clip(np.random.normal(loc=-1, scale=5, size=sum(treatment)), a_min=-10, a_max=max_followup / 2)
df = df.fillna(0)
df["id"] = df.index

df_tv = df.copy()
df_tv = to_long_format(df_tv, "time")
for index, row in df_tv.iterrows():
    if (row["treatment"] == 1) and (row["treatment_start_time"] > 0):
        if row["treatment_start_time"] < row["stop"]:
            df_tv.loc[index, "treatment"] = 0
            df_tv.loc[index, "stop"] == row["treatment_start_time"]

            new_row = pd.DataFrame([row])
            new_row["start"] = row["treatment_start_time"]
            df_tv = pd.concat([df_tv, new_row])
        else:
            df_tv.loc[index, "treatment"] == 0

# Fit Cox model
cph = CoxPHFitter()
cph.fit(df[["event", "time", "treatment", "treatment_start_time"]], duration_col='time', event_col='event')
cph.print_summary()

# Optional: Plot survival curves
cph.plot()
plt.title("Estimated log hazard ratios")
plt.show()

# Fit tv Cox model
cph_tv = CoxTimeVaryingFitter()
cph_tv.fit(df_tv[["event", "start", "stop", "treatment"]], event_col="event", start_col="start", stop_col="stop")
cph_tv.print_summary()

cph_tv.plot()
plt.title("Estimated log hazard ratios")
plt.show()
# %%
