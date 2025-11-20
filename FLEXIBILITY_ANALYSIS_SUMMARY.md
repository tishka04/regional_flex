# Flexible Energy Analysis - Full Year Summary

## Overview
This analysis examines the relative shares of flexible energy delivered across the four French regions (Auvergne-Rhône-Alpes, Nouvelle-Aquitaine, Occitanie, and Provence-Alpes-Côte d'Azur) for a full year.

---

## Key Findings

### Total Flexible Energy: **441,449,813 MWh** (441.4 TWh)

### Relative Shares:

| Category | Energy (MWh) | Energy (TWh) | Share (%) |
|----------|-------------|--------------|-----------|
| **1. Interregional Exchanges** | 70,789,224.70 | 70.8 | **16.04%** |
| **2. Demand Response + Storage** | 4,936,644.44 | 4.9 | **1.12%** |
| **3. Dispatchable Generation** | 365,723,944.26 | 365.7 | **82.85%** |

---

## Detailed Breakdown

### 1. Interregional Exchanges (16.04%)
- **Total Energy**: 70.8 TWh
- Represents power flows between the four regions
- Includes all 12 interconnection paths between regions
- Provides flexibility through spatial balancing of supply and demand

### 2. Demand Response + Storage (1.12%)
- **Total Energy**: 4.9 TWh
  - Demand Response: 4,879,568.21 MWh (98.8% of this category)
  - Storage Discharge: 57,076.23 MWh (1.2% of this category)
    - STEP (Pumped Hydro Storage)
    - Battery Storage
- Smallest contributor to flexible energy
- Includes load curtailment and energy storage systems

### 3. Dispatchable Generation (82.85%)
- **Total Energy**: 365.7 TWh
- **Dominant source** of flexible energy
- Breakdown by generation type:
  - **Hydro**: 268.5 TWh (73.41% of dispatchable)
  - **Nuclear**: 95.2 TWh (26.03% of dispatchable)
  - **Thermal Gas**: 1.8 TWh (0.50% of dispatchable)
  - **Thermal Fuel**: 0.17 TWh (0.05% of dispatchable)
  - **Biofuel**: 0.05 TWh (0.01% of dispatchable)

---

## Key Insights

1. **Dispatchable generation overwhelmingly dominates** flexible energy provision at 82.85%, with hydropower being the primary contributor.

2. **Interregional exchanges play a significant role** at 16.04%, indicating substantial power trading between regions to balance supply and demand.

3. **Demand response and storage have minimal impact** at only 1.12%, suggesting limited deployment or utilization of these technologies during the analysis period.

4. **Within dispatchable generation**:
   - Hydro provides nearly 3/4 of all dispatchable flexibility
   - Nuclear provides about 1/4
   - Thermal, gas, and biofuel contribute minimally (<1% combined)

---

## Implications

- The system relies heavily on traditional dispatchable generation (especially hydro) for flexibility
- There is significant potential to expand demand response and storage capabilities
- Interregional transmission capacity plays an important role in system flexibility
- Future flexibility strategies should consider diversifying beyond hydro dependency

---

## Files Generated

1. `flexibility_shares_summary.csv` - Summary data table
2. `flexibility_shares.png` - Bar and pie chart visualization
3. `flexibility_analysis_detailed.png` - Detailed bar chart with energy values
4. `analyze_flexibility.py` - Analysis script
5. `visualize_flexibility.py` - Visualization script

---

*Analysis completed: Full year data from full_year.csv*
*Four regions analyzed: Auvergne-Rhône-Alpes, Nouvelle-Aquitaine, Occitanie, Provence-Alpes-Côte d'Azur*
