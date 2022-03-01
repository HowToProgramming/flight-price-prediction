from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from numpy import mean, std, sum, sqrt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

dataset_dir = "dataset/"

flight_price_dataset = pd.read_csv(dataset_dir + 'Clean_Dataset.csv')

# Main Idea
"""
- For (most) categorical columns, use binary
- flight ID -> boom
- Model Selection & Colinearity check if possible
- Try linear regression and non linear ones (non linear correlation table or something)
"""


# Drop unnecessary data

flight_price_dataset.drop('Unnamed: 0', inplace=True, axis=1)
flight_price_dataset.drop('flight', inplace=True, axis=1)

# Describe Quantitative Data

print(flight_price_dataset.describe())

# Plot and find the correlation

def plot_and_correlation(column1, column2):
    plt.figure()
    plt.scatter(flight_price_dataset[column1], flight_price_dataset[column2])
    correlation_coef = sum(((flight_price_dataset[column1] - mean(flight_price_dataset[column1])) * (flight_price_dataset[column2] - mean(flight_price_dataset[column2]))))
    correlation_coef /= sqrt(sum((flight_price_dataset[column1] - mean(flight_price_dataset[column1])) ** 2) * sum((flight_price_dataset[column2] - mean(flight_price_dataset[column2])) ** 2))
    return correlation_coef

coef_duration_price = plot_and_correlation('duration', 'price')
print("Correlation Coefficient between duration and price :", coef_duration_price) # 0.20422236784542[redated]

coef_days_price = plot_and_correlation('days_left', 'price')
print("Correlation Coefficient between days_left and price :", coef_days_price) # -0.09194853217143852

# plt.show()

# duration & days are useless pieces of shit

# Print Remaining Columns

print("Available Columns :", list(flight_price_dataset.columns))

# Boxplot each class
plt.cla()
plt.figure()
fig, ax = plt.subplots()
ax.boxplot([flight_price_dataset[flight_price_dataset['class'] == 'Economy']['price'], flight_price_dataset[flight_price_dataset['class'] == 'Business']['price']])
ax.set_xticklabels(['Economy', 'Business'])
plt.savefig('economyvsbusiness.png')

# Hypothesis test ?

# Plot for each airlines
plt.cla()
plt.figure()
fig, ax = plt.subplots()
airlines = list(set(list(flight_price_dataset['airline'])))
price_data = [flight_price_dataset[flight_price_dataset['airline'] == airline]['price'] for airline in airlines]
ax.boxplot(price_data)
ax.set_xticklabels(airlines)
plt.savefig('airlines.png')

# Plot for each Start-destination point
source_city = list(set(list(flight_price_dataset['source_city'])))
dest_city = list(set(list(flight_price_dataset['destination_city'])))

def get_price_source_dest(source, dest):
    return f"{source}-{dest}", flight_price_dataset[(flight_price_dataset['source_city'] == source) & (flight_price_dataset['destination_city'] == dest)]['price']

lbls = []
dataset = []

for source in source_city:
    for dest in dest_city:
        lbl, ds = get_price_source_dest(source, dest)
        if len(ds) == 0:
            continue
        lbls.append(lbl)
        dataset.append(ds)
        
plt.cla()
plt.figure()
fig, ax = plt.subplots()
ax.boxplot(dataset)
ax.set_xticklabels(lbls)
plt.savefig('source-dest.png')

# Plot Stops

plt.cla()
plt.figure()
fig, ax = plt.subplots()
stops = list(set(list(flight_price_dataset['stops'])))
price_data = [flight_price_dataset[flight_price_dataset['stops'] == stop]['price'] for stop in stops]
ax.boxplot(price_data)
ax.set_xticklabels(stops)
plt.savefig('stops.png')

# First Approach : Linear Regression
new_dataset = flight_price_dataset.copy()
new_dataset['business_bin'] = 1
new_dataset['business_bin'][flight_price_dataset['class'] != 'Business'] = 0
for airline in airlines[:-1]:
    new_dataset[airline + "_bin"] = 1
    new_dataset[airline + "_bin"][flight_price_dataset['airline'] != airline] = 0

for stop in stops[:-1]:
    new_dataset[stop + "_bin"] = 1
    new_dataset[stop + "_bin"][flight_price_dataset['stops'] != stop] = 0

for start in source_city[:-1]:
    new_dataset[start + '_bin'] = 1
    new_dataset[start + '_bin'][flight_price_dataset['source_city'] != start] = 0

for start in dest_city[:-1]:
    new_dataset[start + '_bin'] = 1
    new_dataset[start + '_bin'][flight_price_dataset['destination_city'] != start] = 0

keep_columns = list(filter(lambda x: 'bin' in x, list(new_dataset.columns))) + ['duration']

new_dataset = new_dataset[keep_columns]
prices = flight_price_dataset['price']

print(new_dataset)

# Model

model = LinearRegression()
model.fit(new_dataset, prices)
y_hat = model.predict(new_dataset)
coefs, intercept = model.coef_, model.intercept_

print(y_hat)

# Check correlation

r2 = r2_score(prices, y_hat)
adj_r2 = 1 - (1 - r2) * (len(prices) - 1) / (len(prices) - len(new_dataset.columns) - 1) 
print(adj_r2)

flight_price_dataset['y_pred_linear'] = y_hat

# Second Approach : Decision Tree 



clf = DecisionTreeRegressor()
clf.fit(new_dataset, prices)
y_pred = clf.predict(new_dataset)
flight_price_dataset['y_pred_decisiontree'] = y_pred
flight_price_dataset.to_csv('predicted.csv')

r2 = r2_score(prices, y_pred)
adj_r2 = 1 - (1 - r2) * (len(prices) - 1) / (len(prices) - len(new_dataset.columns) - 1) 
print(adj_r2)