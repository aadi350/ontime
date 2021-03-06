This project uses the [Airline Reporting Carrier On-Time Performance Dataset](https://developer.ibm.com/exchanges/data/all/airline/?mhsrc=ibmsearch_a&mhq=%20Airline)
from IBM Developer. This dataset includes two CSVs containing all US Domestic Flight Data from the BTS between 1987 and 2020. The dataset contains basic information about each flight (such as date, time, departure airport, arrival airport) and, if applicable, the amount of time the flight was delayed and information about the reason for the delay.

This dataset included two splits, a randomly sampled split containing 2e06 samples, and a full split containing 2e08 rows. For the purposes of this project, the former dataset will be used

# Project Goals
Predict the likelihood of your flight arriving on time given week-day, scheduled departure time, route and time-of-year (quarterly or weekly). This will be formulated as a classification problem, and will be solved using the Decision-Tree Algorithm. 

# How is this Actually Useful?
An individual is now able to make a data-driven decision to choose the airline with the lowest probability of arriving late for a given route whilst controlling for the parameters above

# Learning Outomes
- [x] Become familiar with Bokeh and Datashader for plotting large datasets
- [x] Use NVIDIA Rapids to accelerate parralelizable operations on the GPU for columnar operations
- [x] Use XGBoost on GPU to perform the aforementioned classification

# Technologies Used
- [NVIDIA RAPIDS](https://developer.nvidia.com/rapids) 
- [Bokeh](https://docs.bokeh.org/en/latest/index.html)
- [Datashader](https://datashader.org/)
- [XGBoost](https://github.com/dmlc/xgboost)

# Future Plans
- Re-do project using Dask and RAPIDS to perform classification on out-of-memory dataframes
- Deployment on a Flask Server to perform real-time prediction of the chances of your flight being delayed


# Tasks
- [x] Explore feature-selection methods (stepwise, lasso, etc)
- [x] Address multicollinearity
- [x] Read/implement Random forest using XGBoost


# Sample Insights

The data-story for this project is located mainly in the [EDA Notebook](src/eda.ipynb)
The following is a static screenshot of a visualisation showing the airports with the most arrival delays during the period 1987-2020
![alt text](img/overall_map.png)

I started exploring the relationship between arrival delay and arrival time, with the (possibly naive) idea that time-of-day corresponds to more/less flights being delayed. However, as shown in the graph below, there did not seem to be any explicit relationship between the two. However, a key idea that seems likely is that the variance of your flight arriving late decreases towards the middle of the day. This may be a by-product of my intial idea, where the centre of the day is more stable owing to periodic, business-related activities, whilst the end of the day is a time of chaos. Additionally, this end-of-day increase in variance may be additionally influenced by international flights arriving and departing.
![Arrival Delay by Time of Day](img/arrdelay_arrtime.png)

When exploring the relationship between arrival delay and time-of-year (after controlling for net number of flights by averaging across years), it seems that the mid-summer period, the Christmas season and the beginning of the year.
![Arrival Delay by Time of Year](img/del_timeofyear.png)

Interestingly, the below airlines seem to have the most flight delays. These percentages were controlled for the number of flights per individual airline, such that the net number of flights arriving late would not affect the overall percentage. In this way, the *proportion* of flights delayed per airline is instead visualised.
![Jetblue is in trouble](img/del_airline.png)

# Training a Regression Tree
The XGBoost Algorithm was used to estimate the chance of your flight arriving late using the parameters as mentioned above. (Overall mean-squared error on a 2-1 train test split was approximately 0.1, which isn't exactly great, but is far better than pure chance)

The below shows the relative importance of features found by the Gradient-boosted tree. The full tree diagram is located in ![src/train.ipynb]

![F-Test](img/featureimportance.png)

# What I'd Have Done Differently
1. I would have explored more ways of measuring the performance of the Gradient-boosted tree, MSE, while informative, is limited in relating feature-importance to actual errors produced
2. More research into what actually causes a delay (from blogs/research papers) would have been done up-front, I dove into this expecting the data to hold all insights, however, this seems to have been a bit misguided owing to the MSE found
