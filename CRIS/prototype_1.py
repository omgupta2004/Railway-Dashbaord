# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# 2. Load and Clean Data
try:
    df = pd.read_csv("C:\\Users\\omgup\\OneDrive\\Desktop\\CRIS\\Quota_wise_demand.csv")
except FileNotFoundError:
    print("Error: 'Quota_wise_demand.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Clean column names
df.columns = df.columns.str.strip(" '\"")
# Clean string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.replace("'", "")

# Convert 'journey_date' to datetime
df['journey_date'] = pd.to_datetime(df['journey_date'], errors='coerce')
print("Data cleaning complete.")

# 3. Exploratory Data Analysis (EDA)

# Total Passenger Demand Over Time
daily_demand = df.groupby('journey_date')['PSGN'].sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_demand.index, daily_demand.values)
plt.title("Total Passenger Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Total Passengers")
plt.grid(True)
plt.show()

# Demand by Quota and Class
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
quota_demand = df.groupby('QUOTA_CODE')['PSGN'].sum().sort_values(ascending=False)
quota_demand.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title("Total Demand by Quota")
axes[0].set_xlabel("Quota")
axes[0].set_ylabel("Total Passengers")
class_demand = df.groupby('cls')['PSGN'].sum().sort_values(ascending=False)
class_demand.plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title("Total Demand by Class")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Total Passengers")
plt.tight_layout()
plt.show()

# Demand by Day of the Week
df['weekday'] = df['journey_date'].dt.day_name()
weekday_demand = df.groupby('weekday')['PSGN'].sum().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
plt.figure(figsize=(10, 6))
weekday_demand.plot(kind='bar', color='purple')
plt.title("Total Demand by Day of the Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Passengers")
plt.xticks(rotation=45)
plt.show()

# Passenger Demand Heatmap (Boarding vs. Destination)
top_brd = df.groupby('brdpt_code')['PSGN'].sum().nlargest(15).index
top_res = df.groupby('resupto_code')['PSGN'].sum().nlargest(15).index
top_df = df[df['brdpt_code'].isin(top_brd) & df['resupto_code'].isin(top_res)]
od_matrix = top_df.pivot_table(values='PSGN', index='brdpt_code', columns='resupto_code', aggfunc='sum', fill_value=0)
plt.figure(figsize=(15, 10))
sns.heatmap(od_matrix, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title("Passenger Demand Heatmap (Top 15 Stations)")
plt.xlabel("Destination Station")
plt.ylabel("Boarding Station")
plt.show()

# 4. Prepare Aggregated Data for Dashboard
print("\n--- Preparing Data for Time-Series ---")
agg_df = df.groupby(["journey_date", "cls", "QUOTA_CODE"], as_index=False)["PSGN"].sum()
agg_df['cls'] = agg_df['cls'].str.strip()
agg_df['QUOTA_CODE'] = agg_df['QUOTA_CODE'].str.strip()
agg_df = agg_df.sort_values("journey_date")
agg_df["dayofweek"] = agg_df["journey_date"].dt.dayofweek
agg_df["month"] = agg_df["journey_date"].dt.month
agg_df["is_weekend"] = agg_df["dayofweek"].isin([5, 6]).astype(int)
le_cls = LabelEncoder()
le_quota = LabelEncoder()
print("Unique classes seen by LabelEncoder:", sorted(agg_df['cls'].unique())) # For debugging
agg_df["cls_enc"] = le_cls.fit_transform(agg_df["cls"])
agg_df["quota_enc"] = le_quota.fit_transform(agg_df["QUOTA_CODE"])
for lag in [1, 2, 3, 7, 14, 30]:
    agg_df[f"lag{lag}"] = agg_df.groupby(["cls_enc", "quota_enc"])["PSGN"].shift(lag)
agg_df["rolling_7_mean"] = agg_df.groupby(["cls_enc", "quota_enc"])["PSGN"].shift(1).rolling(7).mean()
agg_df["rolling_7_std"] = agg_df.groupby(["cls_enc", "quota_enc"])["PSGN"].shift(1).rolling(7).std()
agg_df = agg_df.dropna()
print("Feature engineering complete.")

# 5. Save Aggregated Data for Dashboard
agg_df.to_csv("aggregated_demand_data.csv", index=False)
print("\nAggregated data saved for dashboard use.")
