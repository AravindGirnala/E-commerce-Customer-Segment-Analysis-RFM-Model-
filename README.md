import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
# Note: Ensure you have the dataset 'data.csv' in your directory
df = pd.read_csv('data.csv', encoding="ISO-8859-1")

# 2. Data Cleaning
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 3. Calculate RFM Metrics
now = dt.datetime(2011, 12, 10) # Using a snapshot date for this dataset

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (now - x.max()).days, # Recency
    'InvoiceNo': lambda x: len(x),                # Frequency
    'TotalPrice': lambda x: x.sum()               # Monetary
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 4. Scoring (1-5 Scale)
rfm["R_Score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["F_Score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["M_Score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# 5. Segmenting
def segment_customer(df):
    score = str(df['R_Score']) + str(df['F_Score'])
    if score in ['55', '54', '45']: return 'Champions'
    elif score in ['33', '43', '44']: return 'Loyal'
    elif score in ['51', '52', '41', '42']: return 'New Customers'
    elif score in ['11', '12', '21', '22']: return 'Hibernating/Lost'
    else: return 'At Risk'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# 6. Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='Segment', data=rfm, palette='magma')
plt.title('Customer Segments Distribution')
plt.savefig('visualizations/segments_chart.png')
print("Analysis complete. Segment chart saved.")
