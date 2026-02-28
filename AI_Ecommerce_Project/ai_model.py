import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("ecommerce_sales_34500.csv")

df['returned'] = df['returned'].map({'No': 0, 'Yes': 1})

X = df[['price',
        'discount',
        'quantity',
        'delivery_time_days',
        'shipping_cost',
        'customer_age',
        'profit_margin']]

y = df['returned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

df['return_probability'] = model.predict_proba(X)[:, 1]

df['risk_score'] = df['return_probability'] * df['total_amount']

def generate_suggestion(row):
    
    if row['return_probability'] > 0.7 and row['profit_margin'] < 5:
        return "High Risk & Low Margin – Consider Removing Product"
    
    elif row['return_probability'] > 0.7 and row['delivery_time_days'] > 5:
        return "High Risk – Improve Delivery Speed"
    
    elif row['return_probability'] > 0.7 and row['discount'] > 0.3:
        return "High Risk – Reduce Heavy Discounts"
    
    elif row['return_probability'] > 0.7:
        return "High Risk – Flag for Manual Review"
    
    elif row['return_probability'] > 0.4:
        return "Medium Risk – Monitor Customer Behavior"
    
    else:
        return "Low Risk – Safe Order"

df['AI_Suggestion'] = df.apply(generate_suggestion, axis=1)

def risk_category(prob):
    if prob > 0.7:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"

df['Risk_Level'] = df['return_probability'].apply(risk_category)

df.to_csv("ai_output.csv", index=False)

print("AI Model Completed Successfully!")
print("AI output file created: ai_output.csv")