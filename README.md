# SCT_ML_1
SKILLCRAFT TECHNOLOGY INTERNSHIP
Here's an example of a **README** file for your linear regression house price prediction project. It will explain the structure of the project, how to run the code, and give an overview of the steps and the expected output.

---

# House Price Prediction using Linear Regression

## Project Overview

This project uses **linear regression** to predict house prices based on square footage, number of bedrooms, full bathrooms, and half bathrooms. We use two datasets: one containing the features of the houses (square footage, number of bedrooms, etc.) and the other containing the target variable (house prices). The project demonstrates data preprocessing, splitting data into training and testing sets, training the linear regression model, and evaluating its performance using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)**. 

### The project steps include:
- **Data Loading**: Load datasets containing features and house prices.
- **Data Cleaning**: Handle missing values and select relevant features.
- **Data Splitting**: Split the data into training and testing sets.
- **Model Training**: Train a linear regression model on the training set.
- **Evaluation**: Evaluate the model using various metrics and visualize the results.
- **Visualization**: Create plots for model performance and feature importance.

## Project Structure

```
|-- project_folder/
|   |-- test.csv               # Features dataset
|   |-- sample_submission.csv  # Target dataset (SalePrice)
|   |-- house_price_prediction.ipynb  # Jupyter notebook file
|   |-- README.md              # Project documentation
```

## Requirements

Before running the project, make sure you have the following Python libraries installed:

- `pandas` for data manipulation
- `numpy` for numerical calculations
- `matplotlib` for data visualization
- `scikit-learn` for machine learning models and metrics

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Instructions

### 1. Data Loading

In the first step, the datasets are loaded into two separate DataFrames: 
- `test.csv`: Contains house features (like square footage, number of bedrooms, bathrooms).
- `sample_submission.csv`: Contains the target variable, **SalePrice** (house prices).

These datasets are merged based on the common column **Id**.

```python
# Load the datasets
df_features = pd.read_csv('test.csv')
df_target = pd.read_csv('sample_submission.csv')

# Merge the datasets on 'Id'
df = pd.merge(df_features, df_target, on='Id')
```

### 2. Data Preprocessing

The relevant features such as **GrLivArea** (square footage), **BedroomAbvGr** (number of bedrooms), **FullBath** (full bathrooms), and **HalfBath** (half bathrooms) are selected for the analysis. We also clean the data by checking for missing values and dropping rows that contain them.

```python
# Selecting relevant features and renaming columns for consistency
df = df[['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
df = df.rename(columns={'BedroomAbvGr': 'Bedroom'})
```

### 3. Splitting Data

The dataset is split into two parts: **training data** and **testing data**. The model will be trained on the training data and evaluated on the testing data. The splitting ratio is 80% for training and 20% for testing.

```python
# Split the dataset into training and testing sets (80% train, 20% test)
X = df[['GrLivArea', 'Bedroom', 'FullBath', 'HalfBath']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training

We initialize the **Linear Regression** model and fit it to the training data. After training, the model coefficients and intercept are printed out.

```python
# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)
```

### 5. Model Evaluation

We then make predictions using the test data, calculate performance metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)**, and visualize the results using a scatter plot to compare the actual vs predicted prices.

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted Sale Prices')
plt.show()
```

### 6. Visualization of Residuals and Feature Importance

Additional visualizations include a **residual plot** to check for patterns in prediction errors and a **feature importance plot** to visualize the impact of each feature on house price predictions.

```python
# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Feature importance plot
features = ['GrLivArea', 'Bedroom', 'FullBath', 'HalfBath']
plt.bar(features, model.coef_)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.show()
```

## Expected Output

After running the code, you should see the following outputs:
1. **Model Coefficients and Intercept**: The coefficients represent how much each feature contributes to predicting house prices.
2. **Mean Squared Error (MSE)**: Indicates the average squared difference between the actual and predicted values.
3. **R-squared (R²)**: Measures how well the model explains the variability in the target variable.
4. **Scatter Plot**: Displays a comparison of actual vs predicted prices, showing how well the model performs.
5. **Residual Plot**: Helps identify any patterns in the model's errors, indicating how well the model generalizes to unseen data.
6. **Feature Importance Plot**: Visualizes which features have the greatest impact on the predicted prices.

## Conclusion

This project demonstrates the use of linear regression for predicting house prices based on a set of features. By analyzing the performance metrics and visualizations, you can evaluate how well the model fits the data and make improvements if necessary.
