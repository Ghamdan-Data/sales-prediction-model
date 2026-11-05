import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Historical Data: Advertising Spend vs Actual Sales
data = {
    'Advertising_Spend': [100, 200, 300, 400, 500, 600],
    'Sales': [1500, 2300, 3100, 3900, 4800, 5600]
}
df = pd.DataFrame(data)

# 2. Define Features (X) and Target (y)
X = df[['Advertising_Spend']]
y = df['Sales']

# 3. Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 4. Predict future sales for upcoming budgets ($700 and $800)
future_budgets = pd.DataFrame({'Advertising_Spend': [700, 800]})
predicted_sales = model.predict(future_budgets)

# 5. Display Results
print("--- MACHINE LEARNING SALES PREDICTION REPORT ---")
print(f"Model Coefficient (Growth Rate): {model.coef_[0]:.2f}")
print(f"Model Intercept (Base Sales): {model.intercept_:.2f}")

print("\n--- Sales Forecast for Future Budgets ---")
for spend, pred in zip(future_budgets['Advertising_Spend'], predicted_sales):
    print(f"Ad Budget: ${spend} ---> Predicted Sales: ${pred:,.2f}")
