# sales_data_analysis.py
# Objective: Load and analyze a sales dataset using pandas, visualize with matplotlib.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Set seaborn style for better visuals (matplotlib-based)
sns.set_style("whitegrid")

# Generate synthetic sales dataset
def create_sales_dataset(n_rows=1000):
    """
    Create a synthetic sales dataset.
    Columns: order_id, order_date, product_category, quantity, unit_price, total_price, region.
    """
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_rows)]
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Toys']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'order_id': range(1, n_rows + 1),
        'order_date': np.random.choice(dates, n_rows),
        'product_category': np.random.choice(categories, n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'quantity': np.random.randint(1, 10, n_rows),
        'unit_price': np.random.uniform(5, 200, n_rows).round(2),
        'region': np.random.choice(regions, n_rows)
    }
    data['total_price'] = data['quantity'] * data['unit_price']
    df = pd.DataFrame(data)
    df['order_date'] = pd.to_datetime(df['order_date'])
    # Introduce missing values
    df.loc[np.random.choice(df.index, 10), 'unit_price'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'region'] = np.nan
    return df

# Task 1: Load and Explore the Dataset
print("=== Task 1: Load and Explore the Dataset ===")

# Create and save dataset
csv_file = "sales_data.csv"
try:
    df = create_sales_dataset()
    df.to_csv(csv_file, index=False)
    print(f"Dataset saved as {csv_file}.")
except Exception as e:
    print(f"Error creating dataset: {e}")
    exit()

# Load dataset
try:
    df = pd.read_csv(csv_file)
    df['order_date'] = pd.to_datetime(df['order_date'])
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {csv_file} not found.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (avoid inplace to fix warnings)
try:
    if df.isnull().sum().any():
        print("\nCleaning dataset...")
        df['unit_price'] = df['unit_price'].fillna(df['unit_price'].mean())
        df['total_price'] = df['quantity'] * df['unit_price']
        df['region'] = df['region'].fillna(df['region'].mode()[0])
        print("Missing values handled: unit_price filled with mean, region with mode.")
        print("No missing values remain:")
        print(df.isnull().sum())
    else:
        print("\nNo missing values found. Dataset is clean.")
except Exception as e:
    print(f"Error cleaning dataset: {e}")

# Task 2: Basic Data Analysis
print("\n=== Task 2: Basic Data Analysis ===")

# Compute basic statistics
print("\nBasic Statistics for Numerical Columns:")
print(df[['quantity', 'unit_price', 'total_price']].describe())

# Group by product_category
print("\nMean of Numerical Columns by Product Category:")
print(df.groupby('product_category')[['quantity', 'unit_price', 'total_price']].mean().round(2))

# Task 3: Data Visualization
print("\n=== Task 3: Data Visualization ===")

# Plot 1: Line Chart (Monthly total sales)
print("Creating Line Chart...")
monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['total_price'].sum().reset_index()
monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['order_date'], monthly_sales['total_price'], marker='o', color='b')
plt.title('Total Sales Over Time (2024)')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("line_chart.png")
plt.close()
print("Line Chart saved as line_chart.png")

# Plot 2: Bar Chart (Average total_price by category)
print("Creating Bar Chart...")
plt.figure(figsize=(8, 6))
sns.barplot(x='product_category', y='total_price', data=df, errorbar=None, hue='product_category', palette='Blues', legend=False)
plt.title('Average Sale Value by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Sale Value ($)')
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.close()
print("Bar Chart saved as bar_chart.png")

# Plot 3: Histogram (Total_price distribution)
print("Creating Histogram...")
plt.figure(figsize=(8, 6))
plt.hist(df['total_price'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sale Values')
plt.xlabel('Sale Value ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()
print("Histogram saved as histogram.png")

# Plot 4: Scatter Plot (Unit_price vs. Quantity)
print("Creating Scatter Plot...")
plt.figure(figsize=(8, 6))
plt.scatter(df['unit_price'], df['quantity'], c=df['product_category'].astype('category').cat.codes, 
            cmap='viridis', alpha=0.6)
plt.title('Unit Price vs. Quantity by Product Category')
plt.xlabel('Unit Price ($)')
plt.ylabel('Quantity')
plt.colorbar(label='Product Category (Encoded)')
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()
print("Scatter Plot saved as scatter_plot.png")

# Clean up
if os.path.exists(csv_file):
    os.remove(csv_file)
    print(f"\nCleaned up: {csv_file} removed.")

print("\n=== Analysis Complete ===")
print("Check the following files for visualizations:")
print("- line_chart.png")
print("- bar_chart.png")
print("- histogram.png")
print("- scatter_plot.png")