# **Flight Price Prediction Using Machine Learning**
### Project Overview
This project explores flight-ticket pricing data and builds machine-learning regression models to estimate flight prices from itinerary and booking-related features. The work is documented in the Flight_Prediction.ipynb Jupyter Notebook.

The notebook covers data inspection, exploratory data analysis (EDA), categorical encoding, feature scaling, model training, evaluation, and an example prediction for a new ticket.
Objectives
- Explore the factors associated with flight-ticket prices.
- Clean and prepare the dataset for modelling.
- Visualize price distributions and relationships between price and flight attributes.
- Train and compare regression models.
- Evaluate predictions using regression metrics.
- Estimate the price of a sample flight ticket.

#### Dataset
The notebook loads a cleaned CSV dataset named Flight_Prediction_Clean_Dataset.csv.

The dataset is expected to include columns such as:
- airline
- source_city
- departure_time
- stops
- arrival_time
- destination_city
- class
- duration
- days_left
- price — target variable
The notebook removes the Unnamed: 0 index-like column and the flight column before modelling.

#### Tools and Libraries
- Python
- Jupyter Notebook / Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

### Workflow
1. Load the data and inspect its structure and summary statistics.
2. Clean the data by dropping unused columns, trimming whitespace from categorical values, and converting the stops categories to numeric values.
3. Perform EDA, including:
   - Counts of categorical variables
   - Flight-price distribution
   - Price comparisons by airline, departure time, source city, class, and number of stops
   - Box plots for price spread and potential outliers
   - A correlation heatmap
4. Encode categorical features using LabelEncoder.
5. Separate features and target: price is the target variable (y), and the remaining columns are the input features (X).
6. Split the data into training and testing sets using a 75%/25% split.
7. Scale the features with StandardScaler.
8. Train regression models:
   - Linear Regression
   - Decision Tree Regressor (max_depth=3)
   - Random Forest Regressor (n_estimators=100, random_state=42)
9. Evaluate model predictions using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² score
10. Explore feature importance with ExtraTreesRegressor and plot predicted versus actual prices for the Random Forest model.
11. Try a sample prediction for a new ticket using the trained Random Forest model.
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
