Railway Quota-wise Passenger Demand Analysis
Project Description
This project performs exploratory data analysis (EDA) on railway passenger demand data, focusing on quota-wise and class-wise demand trends over time. The project cleans and visualizes historical data to generate insights such as total passenger demand trends, demand by quota and class, day-of-week demand patterns, and heatmaps of boarding versus destination stations.

Motivation
To provide a clear understanding of historical railway demand patterns without including prediction or forecasting functionality. This facilitates data-driven decisions and further analysis by railway authorities and researchers.

Features
Data cleaning and preprocessing of raw quota-wise demand data

Visualizations of:

Total passenger demand over time

Demand by quota and travel class

Demand patterns by day of the week

Heatmap of passenger flows between boarding and destination stations

Data aggregation and feature engineering for time-series analysis (excluding forecasting)

Export of aggregated clean data for use in dashboards or further analysis

Technologies Used
Python 3

pandas for data manipulation

numpy for numerical operations

matplotlib and seaborn for plotting and visualization

scikit-learn for data preprocessing utilities

Getting Started
Prerequisites
Python 3.7+

Required Python packages:

text
pandas
numpy
matplotlib
seaborn
scikit-learn
Installation and Setup
Clone the repository or download the project files.

Place the dataset file Quota_wise_demand.csv in the project directory.

Install dependencies using pip:

text
pip install pandas numpy matplotlib seaborn scikit-learn
Running the Project
Run the script prototype_1.py to clean the data and generate visualizations:

text
python prototype_1.py
This will display demand analysis plots and save the aggregated data as aggregated_demand_data.csv.

Data Source
The data used is quota-wise railway passenger demand, containing columns such as journey date, travel class, quota code, boarding stations, destination stations, and passenger counts.

Future Work
Integrate interactive dashboards (e.g., with Streamlit or Dash)

Add predictive modeling and forecasting features

Expand analysis to incorporate booking patterns and seasonal effects
