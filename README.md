California Housing Price Estimator & Market Analysis
This project utilizes Machine Learning (Random Forest) to predict the median house values in California districts based on 1990 census data. It provides a robust, interactive tool for real estate analysis, helping identify pricing trends based on geography, demographics, and property characteristics.

Data Sources: Data is based on the California Census dataset, originally published in the StatLib repository (widely popularized by the "Hands-On Machine Learning" book).

Project Structure:
app.py: Contains the Streamlit application code that serves the interactive web interface.
datasets/: Directory containing the raw housing.csv dataset.
notebooks/: Contains the Jupyter Notebook detailing the entire Data Science lifecycle (EDA, feature engineering, and model training).
*.pkl files: Saved serialized versions of the preprocessing pipeline (full_pipeline.pkl) and the trained ML model (random_forest_model.pkl).
Dockerfile: Configuration file for containerizing the application.
requirements.txt: List of Python dependencies required to run the project.

Features
Data Analysis & Feature Engineering: Comprehensive EDA to understand the California housing market.
Geospatial Analysis: Visualized the high correlation between proximity to the ocean, major hubs (Bay Area / Los Angeles), and housing prices.
Feature Creation: Engineered new, highly correlated attributes such as rooms_per_household, bedrooms_per_room, and population_per_household to give the model deeper context beyond raw totals.
Data Cleaning: Handled missing values via median imputation and scaled features to prevent data leakage and bias.

Predictive Modeling: Progression from basic algorithms to an optimized Ensemble model.
Algorithm Selection: Started with Linear Regression and standard Decision Trees, ultimately selecting a Random Forest Regressor to capture complex, non-linear market patterns.
Hyperparameter Tuning: Utilized RandomizedSearchCV to fine-tune the forest (adjusting n_estimators, max_features, etc.), significantly reducing the Root Mean Squared Error (RMSE).
Robust Pipeline: Implemented a ColumnTransformer pipeline to seamlessly encode categorical data (One-Hot Encoding) and standardize numerical data, ensuring production data is treated exactly like training data.

Interactive Web Interface: A user-friendly Streamlit application for real-time property valuation.
Real-time Prediction: Users can adjust sliders for income, house age, and coordinates to instantly see the AI's estimated property value.
Visual Enhancements: Integrated Lottie animations and dynamic geographic mapping to visually place the district being analyzed.
Error Handling: Built-in safeguards to handle data dimension mismatches and ensure smooth inference.

Dockerized: Fully containerized for consistent deployment anywhere.
Portability: Thanks to the Dockerfile and requirements.txt, the application runs in an identical environment on any server.

Tech Stack
Language: Python 3.11
Machine Learning: Scikit-learn, Pandas, NumPy
Web Application: Streamlit, Streamlit-Lottie
Model Persistence: Joblib
DevOps: Docker

Business Insights & Final Results
Based on the final optimized Random Forest model, the following market insights were obtained:

Model Performance:
Error Reduction: The ensemble approach successfully dropped the relative prediction error from ~34% (Linear Regression) down to ~23%.
Generalization: The model demonstrated stable performance on unseen test data, proving its reliability for real-world baseline estimates.

Key Findings:
Income is King: median_income proved to be the single most powerful predictor of housing prices. A small increase in median district income drastically shifts the expected property value.
The Coastal Premium: The categorical feature ocean_proximity (specifically "INLAND" vs. "NEAR OCEAN") plays a massive role. Inland properties suffer a steep discount compared to coastal equivalents, regardless of house age or room count.
Market Capping: The analysis identified artificial price caps in the historical data (at $500,000 for prices and 50 years for age), which highlights the need for careful threshold management in automated valuation models (AVMs).

Proposed Use Cases:
Real Estate Agencies: Rapid baseline valuation for new property listings.
Investment Firms: Identifying potentially undervalued districts by comparing the model's expected price against current market asking prices.

How to Run Locally

Clone the repository and navigate to the folder:

git clone (https://github.com/andriysavcyn/houses-price-prediction.git)
Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
Run the Streamlit App:

streamlit run app.py
Open http://localhost:8501 in your browser.

How to Run with Docker

Build the image:

docker build -t california-house-app .
Run the container:

docker run -p 8501:8501 california-house-app
Open http://localhost:8501 in your browser.