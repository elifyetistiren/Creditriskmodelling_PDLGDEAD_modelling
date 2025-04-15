
# Credit Risk Modelling Project (PD, LGD, EAD, and EL)

This comprehensive project simulates the institutional credit risk modelling workflow, aligning with best practices across global banking and investment institutions. It demonstrates an end-to-end analytical pipeline to evaluate institutional counterparties using publicly available regulatory filings (SEC), simulate internal credit scoring models, estimate forward-looking risk parameters (PD, LGD, EAD), and compute expected losses (EL) as required by Basel regulatory standards.

----------

## 📦 Part 1: Data Preparation and Feature Engineering

### Load and Filter SEC Data

In this foundational step, we extract financial and metadata disclosures from the SEC's structured filing datasets. We focus on publicly listed financial institutions by filtering the metadata file (`sub.txt`) using key SIC (Standard Industrial Classification) codes that correspond to investment banks, brokers, asset managers, holding companies, and related entities. This targeted selection ensures relevance for institutional credit risk modeling.

```python
import pandas as pd

sub = pd.read_csv("sub.txt", sep='	', low_memory=False)
num = pd.read_csv("num.txt", sep='	', low_memory=False)
relevant_sics = [6111, 6211, 6282, 6719, 6726, 6799]
institutions = sub[sub['sic'].isin(relevant_sics)]

```

### Pivot and Merge Numeric Tags

We extract a curated list of financial tags essential for credit analysis — including various revenue, debt, equity, and liability definitions. These tags are extracted from the numeric financials file (`num.txt`). We filter rows that match the institution list and relevant tags, then pivot the resulting long-form data into a wide format, yielding one row per institution with clearly labeled financial attributes.

```python
tags_needed = [
    'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet',
    'AssetsCurrent', 'Assets',
    'LiabilitiesCurrent', 'Liabilities',
    'InterestExpense', 'InterestExpenseOperating', 'InterestAndDebtExpense',
    'LongTermDebt', 'DebtLongtermAndShorttermCombinedAmount',
    'StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    'OperatingIncomeLoss', 'RetainedEarningsAccumulatedDeficit'
]

num_filtered = num[num['adsh'].isin(institutions['adsh']) & num['tag'].isin(tags_needed)]
pivot_df = num_filtered.pivot_table(index='adsh', columns='tag', values='value', aggfunc='first').reset_index()
final_df = pd.merge(institutions[['adsh', 'name', 'sic']], pivot_df, on='adsh')
final_df = final_df.rename(columns={'name': 'Counterparty_Name'})

```

### Handle Missing Financials

Due to variations in reporting standards and periods, some entities may omit key financial metrics. To create a complete and analytically usable dataset, we implement a fallback mechanism using `safe_combine()` to consolidate similar tags. For example, if "Revenues" is missing, we fall back to alternate revenue tags.

This fallback logic is followed by imputation. First, we impute missing values using the median within the same SIC industry group (peer median). If any values remain missing, we impute using the overall median across the dataset. This hierarchical imputation ensures contextual accuracy while avoiding extreme bias from global statistics.

```python
def safe_combine(df, cols):
    valid_cols = [col for col in cols if col in df.columns]
    if not valid_cols:
        return pd.Series([None] * len(df), index=df.index)
    result = df[valid_cols[0]]
    for col in valid_cols[1:]:
        result = result.combine_first(df[col])
    return result

final_df['Revenue'] = safe_combine(final_df, ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'])
final_df['Debt'] = safe_combine(final_df, ['LongTermDebt', 'DebtLongtermAndShorttermCombinedAmount'])
final_df['Equity'] = safe_combine(final_df, ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'])
final_df['Interest_Expense'] = safe_combine(final_df, ['InterestExpense', 'InterestExpenseOperating', 'InterestAndDebtExpense'])
final_df['Current_Assets'] = safe_combine(final_df, ['AssetsCurrent', 'Assets'])
final_df['Current_Liabilities'] = safe_combine(final_df, ['LiabilitiesCurrent', 'Liabilities'])
final_df['Total_Assets'] = safe_combine(final_df, ['Assets'])
final_df['Total_Liabilities'] = safe_combine(final_df, ['Liabilities'])
final_df['Retained_Earnings'] = safe_combine(final_df, ['RetainedEarningsAccumulatedDeficit'])
final_df['Operating_Income'] = safe_combine(final_df, ['OperatingIncomeLoss'])

```

### Imputation and Data Cleansing

After consolidating multiple financial reporting tags, we proceed with imputation to ensure model-ready completeness. We first group by SIC code and fill missing financial values with the peer group's median, preserving industry-specific characteristics. If missing values still persist, we use the global median across all entities as a fallback. This approach balances specificity with coverage, following standard practices in financial modeling.

We also remove entities reporting negative debt balances, as these are often due to data entry or mapping errors and do not reflect valid financial behavior.

```python
# Fill missing financials with SIC-wise medians, then global median
financial_cols = ['Revenue', 'Debt', 'Equity', 'Interest_Expense', 'Current_Assets', 'Current_Liabilities', 'Total_Assets', 'Total_Liabilities', 'Retained_Earnings', 'Operating_Income']
for col in financial_cols:
    final_df[col] = final_df.groupby('sic')[col].transform(lambda x: x.fillna(x.median()))
    final_df[col] = final_df[col].fillna(final_df[col].median())

# Remove companies with negative debt (data issue)
# As a final data cleaning step, we eliminate firms with negative reported debt. 
# Such entries are considered invalid for credit analysis as they violate basic financial logic and often result from misclassifications or submission errors.
final_df = final_df[final_df['Debt'] >= 0]

```

----------

## 🧭 Part 2: Sector Mapping and Financial Signal Engineering

### Assigning Sector Labels

To facilitate sector-specific modeling of credit risk, we enhance each institution's profile with a standardized sector label. This process supports the segmentation of counterparties into analytically meaningful categories, aiding in LGD calibration, exposure allocation, and regulatory aggregation.

Our classification approach follows a dual-pass structure:

1.  **SIC-Based Mapping**: For entities with valid SIC codes, we apply a direct mapping into high-level financial sectors using a predefined dictionary. These include categories such as 'Credit Agency / Investment Bank', 'Broker', and 'Asset Manager'.
    
2.  **Keyword-Based Inference**: For institutions with missing or ambiguous SIC codes, we infer the sector by examining the firm’s legal name. This heuristic approach searches for financial industry terms like “Holding”, “Securities”, “Investment”, or “Fund” to determine the likely business model. This is particularly useful for complex entities such as holding companies or private equity firms that may report under diversified or legacy SIC classifications.
    

This dual method ensures that sector attribution is both comprehensive and context-aware, improving the reliability of downstream modeling.

```python
sic_map = {
    6111: 'Credit Agency / Investment Bank',
    6211: 'Broker',
    6282: 'Asset Manager',
    6719: 'Holding Company',
    6726: 'Investment Office',
    6799: 'Investor / PE'
}

def infer_sector(name):
    name = str(name).upper()
    if 'BROKER' in name or 'SECURITIES' in name:
        return 'Broker'
    elif 'ASSET' in name or 'INVESTMENT' in name or 'FUND' in name:
        return 'Asset Manager'
    elif 'BANK' in name or 'CAPITAL' in name or 'MORTGAGE' in name or 'CREDIT' in name:
        return 'Credit Agency / Investment Bank'
    elif 'HOLDING' in name or 'HOLDINGS' in name:
        return 'Holding Company'
    elif 'PARTNER' in name or 'PARTNERS' in name or 'EQUITY' in name or 'VENTURE' in name:
        return 'Investor / PE'
    return 'Other'

final_df['Sector'] = final_df['sic'].map(sic_map)
final_df['Sector'] = final_df.apply(lambda row: row['Sector'] if pd.notna(row['Sector']) else infer_sector(row['Counterparty_Name']), axis=1)

```


### Part 2: Altman Z-Score Based Credit Risk Signals

The **Altman Z-Score** is a proven financial risk metric used to evaluate the creditworthiness of firms by combining various financial ratios into a single composite index. Originally developed to predict corporate bankruptcy, it remains a valuable tool for assessing the financial distress potential of institutional counterparties, particularly in early warning frameworks.

In this project, we compute a customized Z-Score using the following five financial ratios, each reflecting a distinct dimension of risk:

1. **Liquidity Signal – Working Capital to Total Assets ($Z_1$)**  
   This ratio gauges a firm's ability to meet short-term obligations with short-term assets:  
   $$
   Z_1 = \frac{\mathrm{Current\ Assets} - \mathrm{Current\ Liabilities}}{\mathrm{Total\ Assets}}
   $$

2. **Profitability Retention – Retained Earnings to Total Assets ($Z_2$)**  
   Indicates the firm’s cumulative profitability over time, acting as a proxy for resilience:  
   $$
   Z_2 = \frac{\mathrm{Retained\ Earnings}}{\mathrm{Total\ Assets}}
   $$

3. **Operating Efficiency – EBIT to Total Assets ($Z_3$)**  
   Captures how effectively the company utilizes its assets to generate operating income:  
   $$
   Z_3 = \frac{\mathrm{Operating\ Income}}{\mathrm{Total\ Assets}}
   $$

4. **Leverage Buffer – Net Worth to Total Liabilities ($Z_4$)**  
   Represents solvency and the extent of equity cushion available to absorb losses:  
   $$
   Z_4 = \frac{\mathrm{Total\ Assets} - \mathrm{Total\ Liabilities}}{\mathrm{Total\ Liabilities}}
   $$

5. **Asset Turnover – Revenue to Total Assets ($Z_5$)**  
   Measures operational efficiency and revenue generation capacity relative to asset base:  
   $$
   Z_5 = \frac{\mathrm{Revenue}}{\mathrm{Total\ Assets}}
   $$

Using the standard Altman weights, the composite score is computed as:  
$$
\mathrm{Altman\ Z} = 1.2Z_1 + 1.4Z_2 + 3.3Z_3 + 0.6Z_4 + 1.0Z_5
$$

This score helps classify counterparties into financial health zones:
- **Distress Zone**: \( Z < 1.8 \)
- **Grey Zone**: $( 1.8 \leq Z \leq 3.0 )$
- **Safe Zone**: \( Z > 3.0 \)

These Z-zones are then integrated into the probability of default (PD) blending framework to modulate weight between structural (Altman-based) and statistical (logistic regression) PD estimates based on financial signal strength.



```python
# Compute Altman Z-score components
final_df['Z1'] = (final_df['Current_Assets'] - final_df['Current_Liabilities']) / final_df['Total_Assets']
final_df['Z2'] = final_df['Retained_Earnings'] / final_df['Total_Assets']
final_df['Z3'] = final_df['Operating_Income'] / final_df['Total_Assets']
final_df['Z4'] = (final_df['Total_Assets'] - final_df['Total_Liabilities']) / final_df['Total_Liabilities']
final_df['Z5'] = final_df['Revenue'] / final_df['Total_Assets']

# Calculate final Altman Z-score
final_df['Altman_Z'] = (
    1.2 * final_df['Z1'] +
    1.4 * final_df['Z2'] +
    3.3 * final_df['Z3'] +
    0.6 * final_df['Z4'] +
    1.0 * final_df['Z5']
)

# Assign Z-score risk zone
def zscore_zone(z):
    if z < 1.8:
        return 'distress'
    elif z <= 3.0:
        return 'grey'
    else:
        return 'safe'

final_df['Z_Zone'] = final_df['Altman_Z'].apply(zscore_zone)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Counterparty_ID</th>
      <th>Counterparty_Name</th>
      <th>Revenue</th>
      <th>Debt</th>
      <th>Equity</th>
      <th>Interest_Expense</th>
      <th>Current_Assets</th>
      <th>Current_Liabilities</th>
      <th>Sector</th>
      <th>Z1</th>
      <th>Z2</th>
      <th>Z3</th>
      <th>Z4</th>
      <th>Z5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C001</td>
      <td>FRANKLIN RESOURCES INC</td>
      <td>7.390000e+07</td>
      <td>9.167300e+09</td>
      <td>-4.503000e+08</td>
      <td>2.310000e+07</td>
      <td>3.246450e+10</td>
      <td>1.024270e+10</td>
      <td>Asset Manager</td>
      <td>0.684495</td>
      <td>0.367143</td>
      <td>0.006361</td>
      <td>2.169526</td>
      <td>0.002276</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C002</td>
      <td>FEDERAL NATIONAL MORTGAGE ASSOCIATION FANNIE MAE</td>
      <td>2.431590e+08</td>
      <td>6.878500e+10</td>
      <td>7.768200e+10</td>
      <td>9.087400e+10</td>
      <td>2.040000e+11</td>
      <td>4.255074e+12</td>
      <td>Credit Agency / Investment Bank</td>
      <td>-19.858206</td>
      <td>-0.189338</td>
      <td>0.000213</td>
      <td>-0.952057</td>
      <td>0.001192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C003</td>
      <td>SEI INVESTMENTS CO</td>
      <td>0.000000e+00</td>
      <td>9.970000e+08</td>
      <td>2.131828e+09</td>
      <td>4.460000e+08</td>
      <td>1.698670e+08</td>
      <td>7.485300e+07</td>
      <td>Broker</td>
      <td>0.037704</td>
      <td>0.302613</td>
      <td>0.218945</td>
      <td>4.826677</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C004</td>
      <td>SCHWAB CHARLES CORP</td>
      <td>4.187000e+09</td>
      <td>9.970000e+08</td>
      <td>2.000000e+07</td>
      <td>7.170000e+08</td>
      <td>1.586000e+09</td>
      <td>2.304100e+10</td>
      <td>Broker</td>
      <td>-13.527743</td>
      <td>21.375158</td>
      <td>0.028723</td>
      <td>-0.931166</td>
      <td>2.639975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C005</td>
      <td>RAYMOND JAMES FINANCIAL INC</td>
      <td>8.870000e+08</td>
      <td>9.970000e+08</td>
      <td>1.167300e+10</td>
      <td>4.460000e+08</td>
      <td>2.700000e+07</td>
      <td>7.132500e+10</td>
      <td>Broker</td>
      <td>-2640.666667</td>
      <td>440.518519</td>
      <td>1.687185</td>
      <td>-0.999621</td>
      <td>32.851852</td>
    </tr>
  </tbody>
</table>
</div>

----------


## 🏦 Part 3: Internal Rating

### Rule-Based Internal Counterparty Assessment

In institutional credit risk modeling, internal ratings serve as a foundational layer to segment counterparties by financial strength and assign corresponding exposure limits, capital risk weights, or pricing spreads. These ratings typically feed into broader Expected Loss (EL) or Economic Capital (EC) engines and support early warning signal frameworks.

Here, we simulate an internal risk engine aligned with practices observed in bank credit review committees. Instead of relying solely on statistical PD models, we use a **rules-based financial scoring model**, designed for **non-retail entities** such as brokers, hedge funds, asset managers, and credit intermediaries.

Each counterparty is scored across three core dimensions using interpretable, accounting-based ratios:

1. **Capital Adequacy – Debt-to-Equity Ratio**  
   Indicates financial leverage and buffer capacity. Institutions with higher leverage may face higher funding costs or regulatory constraints.
   $$
   \text{Debt-to-Equity} = \frac{\mathrm{Total\ Debt}}{\mathrm{Equity}}
   $$

2. **Debt Servicing Ability – Interest Coverage Ratio**  
   Measures how comfortably a firm can meet interest obligations from recurring income. Often used by analysts and rating agencies to flag refinancing risk.
   $$
   \text{Interest\ Coverage} = \frac{\mathrm{Revenue}}{\mathrm{Interest\ Expense}}
   $$

3. **Liquidity Buffer – Current Ratio**  
   Evaluates whether the firm can honor short-term liabilities using short-term assets. Especially critical for counterparties exposed to margin calls or redemption risk.
   $$
   \text{Current\ Ratio} = \frac{\mathrm{Current\ Assets}}{\mathrm{Current\ Liabilities}}
   $$

Each ratio is scored based on historical thresholds and industry benchmarks. The sum of the scores determines the final rating band, ranging from high risk (CCC) to investment grade (AAA):

- **0–1 points**: CCC — Likely distressed; low limit or full collateralization needed  
- **2–3 points**: B/BB — Speculative; limit setting requires risk mitigants  
- **4–6 points**: BBB to AAA — Investment-grade; eligible for unsecured lines or high risk appetite

> **Note:** As an override rule, counterparties with negative equity (i.e., book value of equity below zero) are **automatically flagged as CCC-rated**. This reflects technical insolvency and aligns with regulatory stress-testing practices.


The Python logic implements the above framework in a reproducible and auditable manner. Key features include:

- **Data safeguards** for divide-by-zero errors and infinities to maintain robustness across real-world accounting data.
- **Ratio computation pipeline** to extract financial leverage, solvency, and liquidity metrics from cleaned financial statement variables.
- **Threshold-based scoring** logic consistent with industry scorecards used in counterparty credit risk.
- **Override rule**: any counterparty with negative equity is automatically rated CCC, even if its ratios appear otherwise favorable.

```python
import numpy as np

# Safeguard for divide-by-zero
final_df['Equity'] = final_df['Equity'].replace(0, np.nan)

# Calculate ratios
final_df['Debt_to_Equity'] = final_df['Debt'] / final_df['Equity']
final_df['Interest_Coverage'] = final_df['Revenue'] / final_df['Interest_Expense']
final_df['Current_Ratio'] = final_df['Current_Assets'] / final_df['Current_Liabilities']

# Remove rows with infinite or missing ratios
final_df = final_df[np.isfinite(final_df[['Debt_to_Equity', 'Interest_Coverage', 'Current_Ratio']]).all(axis=1)]

# Scoring and mapping to internal rating
def assign_rating(row):
    if row['Debt_to_Equity'] < 0:
        return 'CCC'  # Override: negative equity implies technical insolvency
    score = 0
    if row['Debt_to_Equity'] < 1.5: score += 2
    elif row['Debt_to_Equity'] < 2.5: score += 1
    if row['Interest_Coverage'] > 5: score += 2
    elif row['Interest_Coverage'] > 2: score += 1
    if row['Current_Ratio'] > 1.5: score += 2
    elif row['Current_Ratio'] > 1.0: score += 1
    return ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"][min(score, 6)]

final_df['Internal_Rating'] = final_df.apply(assign_rating, axis=1)
```
---


## 📉 Part 4: Probability of Default (PD) Modeling

### Overview

Probability of Default (PD) estimation is a critical input to Expected Credit Loss (ECL), economic capital, and counterparty credit risk frameworks. In institutional credit modeling, PDs are often derived using multiple layers of signal — combining historical experience, internal ratings, and model-driven early warning indicators.

In this section, we implement a hybrid PD modeling pipeline aligned with industry practices used in banking and asset management institutions. Our design leverages:

- **Mapped PDs** sourced from the Moody’s Default & Recovery Database (DRD), reflecting long-run average default rates per credit rating.
- **Logistic regression-based PDs**, trained using U.S. corporate bankruptcy data from Kaggle, offering a statistical view of default likelihood based on real financials.
- **Altman Z-score zones** as a dynamic weighting framework, modulating reliance on expert vs. statistical views based on financial strength.

This blended architecture provides explainability and model robustness, while supporting forward-looking credit risk insights.

---

### 1. Rating-Based PD Mapping (Moody's DRD)

We start by mapping each counterparty’s internal credit rating (AAA to CCC) to a baseline PD using default frequencies published in the Moody’s Default and Recovery Database (DRD). These mappings represent long-term average one-year default rates observed globally.

```python
rating_pd_map = {
    "AAA": 0.0001, "AA": 0.0002, "A": 0.0005, "BBB": 0.002,
    "BB": 0.01, "B": 0.05, "CCC": 0.20
}
final_df['Mapped_PD'] = final_df['Internal_Rating'].map(rating_pd_map)
```

> These mapped PDs provide a stable anchor and are especially useful for investment-grade counterparties where empirical default events are sparse.

---

### 2. Logistic Regression Using U.S. Bankruptcy Data

To supplement mapped PDs with real-time financial signal modeling, we train a logistic regression model using historical U.S. corporate bankruptcy data published on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction).

The features are derived from the Altman Z-score model, covering five key dimensions of financial health:

- Working Capital to Total Assets (Liquidity)
- Retained Earnings to Total Assets (Profitability Retention)
- EBIT to Total Assets (Operating Performance)
- Net Worth to Liabilities (Solvency Buffer)
- Revenue to Total Assets (Asset Turnover)

```python
from sklearn.linear_model import LogisticRegression

# Load and preprocess bankruptcy dataset
bankruptcy_df = pd.read_csv("american_bankruptcy.csv")
bankruptcy_df['Z1'] = (bankruptcy_df['X1'] - bankruptcy_df['X14']) / bankruptcy_df['X10']
bankruptcy_df['Z2'] = bankruptcy_df['X15'] / bankruptcy_df['X10']
bankruptcy_df['Z3'] = bankruptcy_df['X12'] / bankruptcy_df['X10']
bankruptcy_df['Z4'] = (bankruptcy_df['X10'] - bankruptcy_df['X17']) / bankruptcy_df['X17']
bankruptcy_df['Z5'] = bankruptcy_df['X9'] / bankruptcy_df['X10']

# Prepare training data
features = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
X_train = bankruptcy_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = bankruptcy_df['status_label'].apply(lambda x: 1 if x == 'failed' else 0)

# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)

# Apply model to counterparties
X_counterparty = final_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
final_df['PD_Logistic'] = log_reg.predict_proba(X_counterparty)[:, 1]
```

> Logistic regression provides a flexible, interpretable method to model PD using financial ratios, especially useful for identifying early-stage deterioration.

---

### 3. Altman Z-Score Calculation and Risk Zones

We compute the Altman Z-score for each institutional counterparty as a consolidated measure of financial health. This structural model is widely used across corporate finance and credit risk to signal bankruptcy potential.

The formula used is:

```python
final_df['Altman_Z'] = (
    1.2 * final_df['Z1'] +
    1.4 * final_df['Z2'] +
    3.3 * final_df['Z3'] +
    0.6 * final_df['Z4'] +
    1.0 * final_df['Z5']
)
```

Based on Altman’s original thresholds, we classify each counterparty into a financial health "zone":

```python
def zscore_zone(z):
    if z < 1.8:
        return 'distress'
    elif z <= 3.0:
        return 'grey'
    else:
        return 'safe'

final_df['Z_Zone'] = final_df['Altman_Z'].apply(zscore_zone)
```

- **Distress Zone**: Z < 1.8 — High probability of failure  
- **Grey Zone**: 1.8 ≤ Z ≤ 3.0 — Uncertain, needs deeper analysis  
- **Safe Zone**: Z > 3.0 — Financially sound

---

### 4. Blending Mapped and Modeled PDs by Zone

Finally, we blend the expert-based and model-based PDs using Z-score zone logic. This reflects the principle of **confidence-weighted PD estimation** — trusting rating-based mappings more in strong conditions and relying on model PDs under financial stress.

```python
def weighted_pd(row):
    if row['Z_Zone'] == 'safe':
        return 0.8 * row['Mapped_PD'] + 0.2 * row['PD_Logistic']
    elif row['Z_Zone'] == 'grey':
        return 0.5 * row['Mapped_PD'] + 0.5 * row['PD_Logistic']
    else:  # distress
        return 0.3 * row['Mapped_PD'] + 0.7 * row['PD_Logistic']

final_df['Final_PD'] = final_df.apply(weighted_pd, axis=1)
```

This blended PD framework enables:
- Responsive credit risk estimates that adapt to deterioration
- Seamless integration with internal rating workflows
- Forward-looking PDs suitable for limit monitoring, ECL forecasting, or capital modeling


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Counterparty_ID</th>
      <th>Counterparty_Name</th>
      <th>Altman_Z</th>
      <th>Z_Zone</th>
      <th>Mapped_PD</th>
      <th>PD_Logistic</th>
      <th>Final_PD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>C039</td>
      <td>TRILLER GROUP INC.</td>
      <td>-4.369921</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.588207</td>
      <td>0.471745</td>
    </tr>
    <tr>
      <th>10</th>
      <td>C011</td>
      <td>GOLDMAN SACHS GROUP INC</td>
      <td>-678.068327</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.557811</td>
      <td>0.450467</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C010</td>
      <td>OPPENHEIMER HOLDINGS INC</td>
      <td>-2579.017325</td>
      <td>distress</td>
      <td>0.01</td>
      <td>0.637765</td>
      <td>0.449435</td>
    </tr>
    <tr>
      <th>48</th>
      <td>C048</td>
      <td>FEDERAL HOME LOAN BANK OF SAN FRANCISCO</td>
      <td>-0.017571</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.526945</td>
      <td>0.428861</td>
    </tr>
    <tr>
      <th>49</th>
      <td>C049</td>
      <td>FEDERAL HOME LOAN BANK OF TOPEKA</td>
      <td>0.126492</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.525647</td>
      <td>0.427953</td>
    </tr>
    <tr>
      <th>71</th>
      <td>C071</td>
      <td>FEDERAL HOME LOAN BANK OF NEW YORK</td>
      <td>0.141466</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.525486</td>
      <td>0.427840</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C009</td>
      <td>FEDERAL AGRICULTURAL MORTGAGE CORP</td>
      <td>0.246106</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.524915</td>
      <td>0.427440</td>
    </tr>
    <tr>
      <th>80</th>
      <td>C080</td>
      <td>ROBINHOOD MARKETS, INC.</td>
      <td>0.985795</td>
      <td>distress</td>
      <td>0.20</td>
      <td>0.509104</td>
      <td>0.416373</td>
    </tr>
    <tr>
      <th>14</th>
      <td>C015</td>
      <td>MORGAN STANLEY</td>
      <td>-20.435926</td>
      <td>distress</td>
      <td>0.05</td>
      <td>0.538643</td>
      <td>0.392050</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C005</td>
      <td>RAYMOND JAMES FINANCIAL INC</td>
      <td>-2514.254284</td>
      <td>distress</td>
      <td>0.01</td>
      <td>0.548658</td>
      <td>0.387060</td>
    </tr>
  </tbody>
</table>
</div>



----------


## 🧮 Part 5: Loss Given Default (LGD) Estimation

### Institutional LGD Modeling Overview

Loss Given Default (LGD) represents the percentage of an exposure that a lender expects to lose if a counterparty defaults, after accounting for recoveries through collateral liquidation, seniority protections, or restructuring processes.

For institutional portfolios (e.g., brokers, asset managers, hedge funds), LGD is influenced by complex capital structures and collateral arrangements. In this section, we simulate a realistic LGD estimation framework that mirrors internal credit modeling practices across commercial banks and investment firms.

We capture key determinants of LGD through the following drivers:

- **Product type**: Loans, bonds, repos, and derivatives each carry different recovery expectations.
- **Collateral type and haircut**: The quality and liquidity of posted collateral directly affect recovery value.
- **Seniority ranking**: Senior secured exposures typically recover more in a resolution event.
- **Collateral Coverage Ratio (CCR)**: Net collateral relative to exposure provides a structural cushion.
- **Sector-specific risk**: Recovery expectations differ across industries based on historical enforcement data.

---

### Simulating Institutional Exposure Structures

We generate a synthetic exposure portfolio by assigning random but realistic attributes to counterparties, including product type, collateral form, seniority level, and recovery lag assumptions. This enables us to test LGD behavior under varied structural scenarios.

```python
import numpy as np

product_types = ['Loan', 'Bond', 'Repo', 'Derivative', 'Credit Card', 'Line of Credit']
collateral_types = ['Gov Bonds', 'Corporate Bonds', 'Cash', 'Real Estate', 'None']
seniority_levels = ['Senior Secured', 'Senior Unsecured', 'Subordinated']

n = len(final_df)
np.random.seed(42)
final_df['Exposure_Amount'] = np.random.uniform(1e6, 20e6, n).round(2)
final_df['Product_Type'] = np.random.choice(product_types, n)
final_df['Collateral_Type'] = np.random.choice(collateral_types, n)
final_df['Collateral_Value'] = np.random.uniform(0, 20e6, n).round(2)

# Assign collateral haircuts
final_df['Haircut_%'] = np.where(
    final_df['Collateral_Type'] == 'None',
    1.0,
    np.random.uniform(0.05, 0.4, n).round(2)
)

final_df['Seniority'] = np.random.choice(seniority_levels, n)

# Calculate net collateral and collateral coverage ratio
final_df['Net_Collateral'] = final_df['Collateral_Value'] * (1 - final_df['Haircut_%'])
final_df['CCR'] = final_df['Net_Collateral'] / final_df['Exposure_Amount']
```

---

### LGD Estimation Functions

We apply a base LGD assumption based on product category and adjust it using additive logic based on exposure features. This flexible additive approach aligns with credit portfolio models and internal rating-based (IRB) frameworks under Basel.

#### Base LGD by Product Type

Each instrument is assigned a baseline LGD reflecting expected recovery in absence of mitigating features:

```python
def base_lgd_from_product(product):
    return {
        'Loan': 0.45,         # Typical for senior unsecured loans
        'Bond': 0.60,         # Higher due to subordination
        'Repo': 0.08,         # Fully collateralized
        'Derivative': 0.15,   # Netting and CSA reduce loss
        'Credit Card': 0.90,  # Unsecured retail
        'Line of Credit': 0.85
    }.get(product, 0.50)
```

#### Adjustment Factors

Each adjustment adds or subtracts based on exposure protections or risk amplifiers:

```python
def adjust_lgd_from_collateral(collateral):
    return {
        'Cash': -0.05,
        'Gov Bonds': -0.04,
        'Corporate Bonds': -0.02,
        'Real Estate': 0.0,
        'None': 0.20
    }.get(collateral, 0.0)

def adjust_lgd_from_seniority(level):
    return {
        'Senior Secured': -0.10,
        'Senior Unsecured': 0.0,
        'Subordinated': 0.10
    }.get(level, 0.0)

def adjust_lgd_from_ccr(ccr):
    if ccr >= 1.0: return -0.15
    elif ccr >= 0.75: return -0.10
    elif ccr >= 0.5: return -0.05
    elif ccr >= 0.25: return 0.0
    else: return 0.10

def adjust_lgd_from_sector(sector):
    if "bank" in sector.lower(): return -0.05
    elif "hedge" in sector.lower(): return 0.10
    elif "asset manager" in sector.lower(): return 0.05
    elif "broker" in sector.lower(): return 0.0
    else: return 0.0
```

---

### Institutional LGD Calculation

We sum the base LGD with all applicable adjustments and cap the value between 0 and 1, as required in regulatory models.

```python
final_df['LGD_Institutional_Enhanced'] = final_df.apply(
    lambda row: min(
        max(
            base_lgd_from_product(row['Product_Type']) +
            adjust_lgd_from_collateral(row['Collateral_Type']) +
            adjust_lgd_from_seniority(row['Seniority']) +
            adjust_lgd_from_ccr(row['CCR']) +
            adjust_lgd_from_sector(row['Sector']),
        0.0),
    1.0),
axis=1)
```

This modular, explainable LGD engine supports exposure-level risk estimation while maintaining alignment with IRB standards and supervisory LGD benchmarking guidelines.

----------


## 💳 Part 6: Exposure at Default (EAD)

### EAD Framework in Institutional Risk Modeling

Exposure at Default (EAD) represents the expected outstanding exposure to a counterparty at the moment of default. It accounts for:

- **On-balance sheet drawdowns** (e.g., loans, bonds)
- **Off-balance sheet undrawn commitments** (e.g., credit lines, revolvers)
- **Potential future exposure** from products with uncertain exposure profiles (e.g., derivatives, repos)

Our approach reflects real-world credit modeling by assigning each exposure to a **regulatory exposure category** and applying appropriate modeling logic — amortization for term loans, Credit Conversion Factors (CCFs) for revolving products, and SACCR-style formulas for derivative-type exposures.

---

### Step 1: Assign Exposure Category and CCF

We classify each product into exposure categories used in regulatory models:

- **Term**: Fixed-schedule repayments (Loans, Bonds)
- **Revolving**: On-demand access (Lines of Credit, Credit Cards)
- **Other**: Products with mark-to-market or collateralized exposure (Repos, Derivatives)

Each category is assigned a **Credit Conversion Factor (CCF)** to estimate the undrawn exposure likely to be utilized by the time of default.

```python
exposure_type_map = {
    'Loan': 'Term', 'Bond': 'Term',
    'Repo': 'Other', 'Derivative': 'Other',
    'Credit Card': 'Revolving', 'Line of Credit': 'Revolving'
}
final_df['Exposure_Category'] = final_df['Product_Type'].map(exposure_type_map).fillna('Other')

# CCF assignment
final_df['CCF'] = final_df['Exposure_Category'].map({
    'Revolving': 0.75,
    'Term': 1.0
}).fillna(1.0)

# Simulate undrawn limit for revolving exposures
final_df['Undrawn_Limit'] = np.where(
    final_df['Exposure_Category'] == 'Revolving',
    0.25 * final_df['Total_Assets'].fillna(0),
    0
)
```

> Note: The 75% CCF is aligned with Basel’s Standardized Approach for unsecured retail exposures. Term exposures are fully drawn by nature (CCF = 100%).

---

### Step 2: Amortized EAD for Term Exposures

For fixed repayment structures like term loans or bonds, we simulate EAD based on remaining principal at the estimated time of default, using annuity amortization logic.

```python
# Amortization functions
def monthly_payment(principal, r, n):
    return (principal * r * (1 + r)**n) / ((1 + r)**n - 1) if principal > 0 and r > 0 else 0

def remaining_principal(p, r, n, t):
    return p * ((1 + r)**n - (1 + r)**t) / ((1 + r)**n - 1) if p > 0 and r > 0 else 0

# Parameters
monthly_rate = 0.06 / 12
loan_term_months = 60

# Simulate time to default and calculate amortized EAD
final_df['Monthly_Installment'] = final_df['Exposure_Amount'].apply(
    lambda x: monthly_payment(x, monthly_rate, loan_term_months)
)
final_df['Time_to_Default'] = final_df['Final_PD'].apply(
    lambda pd: np.random.randint(1, min(loan_term_months, int((1 - pd) * loan_term_months)) + 1)
)
final_df['EAD_Term_Amortized'] = final_df.apply(
    lambda row: remaining_principal(row['Exposure_Amount'], monthly_rate, loan_term_months, row['Time_to_Default'])
    if row['Exposure_Category'] == 'Term' else 0,
    axis=1
)
```

> This approach mimics internal exposure calculations used in credit limit engines and IFRS 9 lifetime expected loss modeling for amortizing assets.

---

### Step 3: SACCR Exposure for Derivative and Repo Products

For exposures like derivatives and repos, where exposure varies with market movement and collateralization, we apply a **Simplified SACCR (Standardized Approach to Counterparty Credit Risk)** methodology.

```python
def calculate_saccr_ead(row):
    alpha = 1.4  # Regulatory multiplier
    rc = max(row['Exposure_Amount'] - row['Net_Collateral'], 0)
    pfe = row['Exposure_Amount'] * row['Haircut_%']
    return alpha * (rc + pfe)

final_df['EAD_SACCR'] = final_df.apply(
    lambda row: calculate_saccr_ead(row) if row['Exposure_Category'] == 'Other' else 0,
    axis=1
)
```

> **Note:** We use `Haircut_%` to adjust the fair value of posted collateral when computing `Net_Collateral`. This haircut simulates credit risk-adjusted recovery values and reflects regulatory margining and collateral haircut standards under Basel III.  
> $$ \text{Net Collateral} = \text{Collateral Value} \times (1 - \text{Haircut \%}) $$

---

### Step 4: Final EAD Assignment

We consolidate all EAD logic into a unified field, selecting the appropriate methodology per exposure type:

```python
final_df['EAD'] = np.where(
    final_df['Exposure_Category'] == 'Term',
    final_df['EAD_Term_Amortized'],
    np.where(
        final_df['Exposure_Category'] == 'Revolving',
        final_df['Exposure_Amount'] + final_df['CCF'] * (final_df['Undrawn_Limit'] - final_df['Exposure_Amount']),
        final_df['EAD_SACCR']
    )
)
```

This tiered modeling approach ensures:
- Realistic EAD estimates for amortizing structures
- Risk-sensitive exposure recognition for derivatives and repos
- Dynamic undrawn usage assumptions for revolving products


## 🧾 Part 7: Compute Floored Expected Loss

### Applying Regulatory Floors to PD and LGD

In credit risk modeling, regulatory guidelines (e.g., Basel III/IV, IFRS 9) require the application of **minimum floor values** to Probability of Default (PD) and Loss Given Default (LGD). These **floors act as safeguards** against overly optimistic risk assessments, particularly for low-default portfolios or counterparties with incomplete historical data.

Typical floor values:
- **PD Floor**: 0.05% or 0.0005 (reflects minimum risk even for highly rated entities)
- **LGD Floor**: 10% or 0.10 (ensures minimum loss assumption even with strong collateral)

By enforcing floors, institutions reduce the risk of **understated capital requirements** or **insufficient loan loss provisions**.

---

### Step 1: Apply Floors to PD and LGD

```python
final_df['Final_PD_Floored'] = final_df['Final_PD'].apply(lambda x: max(x, 0.0005))
final_df['LGD_Enhanced_Floored'] = final_df['LGD_Institutional_Enhanced'].apply(lambda x: max(x, 0.10))
```

- The floored PD ensures a minimum expected probability of default for every counterparty.
- The floored LGD guarantees a baseline level of potential loss, even for over-collateralized or secured exposures.

---

### Step 2: Compute Expected Loss with Floors

We calculate the final expected loss (EL) incorporating the floored values:

```python
final_df['Expected_Loss_Floored'] = (
    final_df['Final_PD_Floored'] * final_df['LGD_Enhanced_Floored'] * final_df['EAD']
).round(2)
```

The general expected loss formula remains:

$$
\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{EAD}
$$

But here we substitute:
- $\text{PD} \rightarrow\max(\text{PD}, 0.0005)$
- $\text{LGD} \rightarrow\max(\text{LGD}, 0.10)$



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adsh</th>
      <th>Counterparty_Name</th>
      <th>sic</th>
      <th>Assets</th>
      <th>AssetsCurrent</th>
      <th>DebtLongtermAndShorttermCombinedAmount</th>
      <th>InterestAndDebtExpense</th>
      <th>InterestExpense</th>
      <th>InterestExpenseOperating</th>
      <th>Liabilities</th>
      <th>...</th>
      <th>Undrawn_Limit</th>
      <th>Monthly_Installment</th>
      <th>Time_to_Default</th>
      <th>EAD_Term_Amortized</th>
      <th>EAD_SACCR</th>
      <th>EAD</th>
      <th>Expected_Loss</th>
      <th>Final_PD_Floored</th>
      <th>LGD_Enhanced_Floored</th>
      <th>Expected_Loss_Floored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000038777-25-000017</td>
      <td>FRANKLIN RESOURCES INC</td>
      <td>6282.0</td>
      <td>3.246450e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23100000.0</td>
      <td>NaN</td>
      <td>1.024270e+10</td>
      <td>...</td>
      <td>8.116125e+09</td>
      <td>156910.087433</td>
      <td>39</td>
      <td>0.000000e+00</td>
      <td>0.00</td>
      <td>6.089123e+09</td>
      <td>2.130521e+09</td>
      <td>0.349890</td>
      <td>1.00</td>
      <td>2.130521e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000310522-25-000199</td>
      <td>FEDERAL NATIONAL MORTGAGE ASSOCIATION FANNIE MAE</td>
      <td>6111.0</td>
      <td>2.040000e+11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.087400e+10</td>
      <td>4.255074e+12</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>368552.250438</td>
      <td>29</td>
      <td>0.000000e+00</td>
      <td>24610203.65</td>
      <td>2.461020e+07</td>
      <td>0.000000e+00</td>
      <td>0.379767</td>
      <td>0.10</td>
      <td>9.346137e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000350894-25-000028</td>
      <td>SEI INVESTMENTS CO</td>
      <td>6211.0</td>
      <td>2.520003e+09</td>
      <td>169867000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.324940e+08</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>288211.179802</td>
      <td>15</td>
      <td>0.000000e+00</td>
      <td>18451810.46</td>
      <td>1.845181e+07</td>
      <td>3.986148e+05</td>
      <td>0.093926</td>
      <td>0.23</td>
      <td>3.986148e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000316709-25-000010</td>
      <td>SCHWAB CHARLES CORP</td>
      <td>6211.0</td>
      <td>1.586000e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.170000e+08</td>
      <td>2.304100e+10</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>239233.969053</td>
      <td>43</td>
      <td>3.889617e+06</td>
      <td>0.00</td>
      <td>3.889617e+06</td>
      <td>1.610329e+05</td>
      <td>0.115002</td>
      <td>0.36</td>
      <td>1.610329e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0000720005-25-000025</td>
      <td>RAYMOND JAMES FINANCIAL INC</td>
      <td>6211.0</td>
      <td>2.700000e+07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.460000e+08</td>
      <td>7.132500e+10</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>76642.072361</td>
      <td>8</td>
      <td>0.000000e+00</td>
      <td>2164537.38</td>
      <td>2.164537e+06</td>
      <td>0.000000e+00</td>
      <td>0.387060</td>
      <td>0.10</td>
      <td>8.378067e+04</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>


---

### Why Floors Matter

- **Regulatory Compliance**: Basel minimums must be honored in IRB and standardized approaches.
- **Model Robustness**: Floors protect against data sparsity or temporary financial improvements that mask deeper risks.
- **Conservatism in Stress Scenarios**: Helps in avoiding underestimation of losses in economic downturns.

This floored EL serves as the final risk measure that can be used for:
- Pricing adjustments
- Credit reserve planning
- Counterparty-level limit frameworks
- Economic capital allocation

---

## 📊 Part 8: Credit Risk Dashboard Visualization

### Translating Risk Analytics into Business Insights

To support decision-making across risk, finance, and compliance teams, it is essential to present credit model outputs through **intuitive and interactive visualizations**. Using the modeled outputs from our pipeline, we developed a Power BI-style dashboard summarizing institutional credit risk metrics across counterparties, ratings, and sectors.

This step reflects real-world credit risk governance practices, where model outputs are regularly presented to credit committees, senior management, and regulators in dashboard form. The dashboard leverages the final output, capturing fields such as internal ratings, sector classification, and floored expected losses.

---

### Dashboard Layout and Insights

Here is a professional Power BI-style dashboard visualizing key credit risk insights from your `df_sample.csv`:

- **Top-Left**: Distribution of internal ratings assigned to counterparties  
  *Insight*: Identifies portfolio quality and the prevalence of speculative vs. investment-grade ratings.

- **Top-Right**: Sector concentration by number of entities  
  *Insight*: Reveals exposures to specific financial sectors, helping identify concentration risk.

- **Bottom-Left**: Total expected loss aggregated by internal rating  
  *Insight*: Highlights which risk grades contribute most to expected losses, guiding pricing or provisioning decisions.

- **Bottom-Right**: Total expected loss by sector  
  *Insight*: Enables sectoral stress testing, credit appetite planning, and regulatory reporting segmentation.

---

This dashboard converts analytical outputs into **actionable insights** for risk teams, providing an executive-level view of where the credit risk resides — by **sector**, **rating**, and **expected loss**. It enables scenario analysis, credit limit calibration, and capital allocation strategies, all essential to sound institutional credit risk management.


## ✅ Conclusion: Institutional Credit Risk Analytics Framework

This project delivers an end-to-end simulation of an institutional credit risk modeling pipeline, mirroring the methodologies and regulatory expectations applied by global banks, asset managers, and financial risk consultancies.

By leveraging publicly available financial data (SEC EDGAR), the project replicates how financial institutions process counterparties through:

- **Sector classification and financial normalization**
- **Internal rating frameworks aligned with rating committee logic**
- **Modeling Probability of Default (PD) estimation using Moody’s DRD mappings and Altman-based logistic regression models**
- **Loss Given Default (LGD) estimation incorporating product type, collateral quality, seniority structure, and sector risk**
- **Exposure at Default (EAD) simulation across term, revolving, and derivative-style exposures**
- **Application of regulatory floor assumptions on PD and LGD to ensure conservative loss forecasting**

Each component integrates seamlessly into the Expected Loss (EL) formula:

$$
\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{EAD}
$$

This modular framework enables realistic simulation of a counterparty risk lifecycle—from raw financial statement ingestion to the production of credit-sensitive expected loss estimates. It is well-suited for:

- Internal risk rating system prototyping
- Model governance documentation
- Capital planning and RWA simulation
- Portfolio limit calibration or stress testing exercises



By combining structured financial inputs, explainable scorecard models, and industry-anchored assumptions, the project offers a practical, transparent, and regulation-aligned approach to institutional credit risk modeling—one that reflects both analytical rigor and the complexity of modern counterparties.

