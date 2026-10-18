import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("====================================================")
# Context Heading required by project brief
print(" CONTEXT: E-COMMERCE SALES TREND & SEGMENTATION ANALYSIS ")
print("====================================================\n")

# --- STEP 1: CREATE REPLICABLE PRACTICE DATASET ---
rng = np.random.default_rng(42)
n = 2000
orders = pd.DataFrame({
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "customer_id": rng.integers(1, 301, n),
    "category": rng.choice(["Electronics", "Fashion", "Home", "Books"], n, p=[0.3, 0.35, 0.2, 0.15]),
    "amount": rng.gamma(2.0, 600, n).round(2)
})

# --- STEP 2: FEATURE ENGINEERING & DATE EXTRACTION ---
orders["month"] = orders["order_date"].dt.to_period("M")
orders["weekday"] = orders["order_date"].dt.day_name()
orders["quarter"] = orders["order_date"].dt.to_period("Q")

# --- STEP 3: MONTHLY TREND & GROWTH CALCULATIONS ---
monthly_sales = orders.groupby("month")["amount"].sum()
monthly_growth = (monthly_sales.pct_change() * 100).round(1)

best_month = monthly_sales.idxmax()
worst_month = monthly_sales.idxmin()

# --- STEP 4: PLOT TREND LINE CHART ---
plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_sales.index.astype(str), y=monthly_sales.values, marker="o", color="teal", linewidth=2.5)
plt.title("2025 Monthly Revenue Trend Overview", fontsize=12, fontweight="bold")
plt.xlabel("Timeline (Months)")
plt.ylabel("Total Revenue (USD)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)

# Graph annotations highlighting the best and worst months visually
plt.annotate(f"Best Month ({best_month})", xy=(str(best_month), monthly_sales[best_month]), 
             xytext=(str(best_month), monthly_sales[best_month] + 50000),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6))
plt.annotate(f"Worst Month ({worst_month})", xy=(str(worst_month), monthly_sales[worst_month]), 
             xytext=(str(worst_month), monthly_sales[worst_month] - 80000),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

plt.tight_layout()
chart_filename = "sales_trend_chart.png"
plt.savefig(chart_filename, dpi=150)
plt.close()

# --- STEP 5: COMPARE CATEGORIES (GROUPBY & PIVOT TABLE) ---
category_summary = orders.groupby("category")["amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
monthly_category_pivot = pd.pivot_table(orders, values="amount", index="month", columns="category", aggfunc="sum")

# --- STEP 6: FIND YOUR BEST CUSTOMERS (80/20 ANALYSIS) ---
customer_spend = orders.groupby("customer_id")["amount"].sum().sort_values(ascending=False)
top_10_percent_count = int(len(customer_spend) * 0.10)
top_10_spend = customer_spend.head(top_10_percent_count).sum()
total_spend = customer_spend.sum()
top_share_pct = (top_10_spend / total_spend) * 100

# --- PRINT STRUCTURED BUSINESS BRIEF RESULTS TO CONSOLE ---
print("====================================================")
print("                  KEY FINDINGS                      ")
print("====================================================")
print(f"• Total Dataset Shape: {orders.shape[0]} orders collected successfully.")
print(f"• Revenue Peak: Highest sales occurred in {best_month} (${monthly_sales[best_month]:,.2f}).")
print(f"• Revenue Trough: Lowest sales occurred in {worst_month} (${monthly_sales[worst_month]:,.2f}).")
print("\n--- Month-over-Month Growth Rates (%) ---")
for m, growth in zip(monthly_sales.index[1:], monthly_growth[1:]):
    print(f"  {m}: {growth:+.1f}%")

print("\n--- Sales Performance by Product Category ---")
print(category_summary.round(1))

print("\n--- Top 10% Customer Concentration (80/20 Rule Test) ---")
print(f"• The top 10% of premium shoppers bring in {top_share_pct:.1f}% of total enterprise revenue.")

print("\n====================================================")
print("         RECOMMENDATIONS & NEXT STEPS               ")
print("====================================================")
print(f"1. RETAIN PREMIUM SHARE: Launch a dedicated VIP loyalty system for the top 10% cohort who manage {top_share_pct:.1f}% of entire sales volumes.")
print(f"2. CAPITALIZE ON POPULARITY: Scale up catalog inventory allocations for the '{category_summary.index[0]}' category which dominates aggregate gross revenue earnings.")
print(f"3. STABILIZE CORRECTION PERIODS: Prepare counter-promotional bundle strategies ahead of the '{worst_month}' seasonal down-cycle period.")
