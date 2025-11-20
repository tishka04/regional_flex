# Congestion Rent Analysis - Full Year Report

## Executive Summary

**Total Congestion Rent: 170.97 M€**

Congestion rents represent the economic value captured from price differences between interconnected regions. Using the formula **CR_{i→j}(t) = F_{i→j}(t) × (P_j(t) - P_i(t))**, we calculated the total congestion rents for all interregional flows over the full year.

---

## Key Findings

### 1. Regional Congestion Rents (by Exports)

| Region | Abbr. | Congestion Rent (M€) | Share |
|--------|-------|---------------------|-------|
| **Auvergne-Rhône-Alpes** | ARA | **153.63** | 89.9% |
| **Nouvelle-Aquitaine** | NAQ | **21.62** | 12.7% |
| **Occitanie** | OCC | **-4.39** | -2.6% |
| **Provence-Alpes-Côte d'Azur** | PAC | **0.11** | 0.1% |
| **TOTAL** | - | **170.97** | 100% |

**Key Insight**: Auvergne-Rhône-Alpes dominates congestion rent capture, accounting for nearly 90% of the total. This reflects its role as the primary exporter to higher-price regions.

### 2. Top Directional Flows by Congestion Rent

| Flow | Total Flow (MWh) | Avg Price Diff (€/MWh) | Congestion Rent (M€) |
|------|------------------|----------------------|---------------------|
| **ARA → PAC** | 16,725,777.86 | +15.48 | **148.90** |
| **NAQ → PAC** | 2,298,316.88 | +5.08 | **22.27** |
| **ARA → OCC** | 193,549.26 | +7.90 | **4.28** |
| **ARA → NAQ** | 23,631,467.31 | +10.41 | **0.44** |
| **NAQ → ARA** | 24,791,019.34 | -10.41 | **-0.41** |

**Key Insight**: The ARA→PAC flow generates 87% of all congestion rents, with a large flow volume (16.7 TWh) and significant price differential (+15.48 €/MWh).

### 3. Net Congestion Rents by Region Pair

| Region Pair | CR A→B (M€) | CR B→A (M€) | Net CR (M€) |
|-------------|-------------|-------------|-------------|
| **ARA ↔ PAC** | 148.90 | -0.08 | **148.82** |
| **NAQ ↔ PAC** | 22.27 | 0.19 | **22.46** |
| **ARA ↔ OCC** | 4.28 | 0.14 | **4.42** |
| **ARA ↔ NAQ** | 0.44 | -0.41 | **0.03** |
| **NAQ ↔ OCC** | -0.23 | -1.08 | **-1.32** |
| **OCC ↔ PAC** | -3.44 | -0.00 | **-3.44** |

**Key Insight**: The ARA↔PAC pair generates 87% of net congestion rents. Some flows have negative congestion rents when power flows against price gradients.

---

## Detailed Analysis

### Price Structure

**Average Regional Prices (€/MWh):**
- ARA: 172.31 €/MWh (lowest)
- OCC: 180.21 €/MWh
- NAQ: 182.72 €/MWh
- PAC: 187.79 €/MWh (highest)

The price gradient flows from west/center (ARA) to south/east (PAC), with PAC having the highest average price (+15.48 €/MWh above ARA).

### Flow Patterns and Congestion Rents

**Positive Congestion Rents** (flows from low to high price regions):
- ARA exports to all regions generate positive rents
- NAQ exports to PAC generate significant positive rents
- These represent economically efficient power transfers

**Negative Congestion Rents** (flows against price gradients):
- OCC→PAC: -3.44 M€
- OCC→NAQ: -1.08 M€
- These may indicate:
  - Must-run constraints
  - Transmission loop flows
  - Operating reserve requirements
  - Other technical constraints

### Economic Interpretation

#### Total Value: 170.97 M€/year

This represents:
1. **Market efficiency gains** from interregional trade
2. **Value of transmission infrastructure** enabling arbitrage
3. **Potential TSO revenue** if congestion rents are collected
4. **Indicator of transmission constraints** - higher rents suggest valuable but constrained capacity

#### Regional Winners and Losers

**Net Exporters with Positive Rents:**
- **ARA**: 153.63 M€ - Dominant exporter to high-price regions
- **NAQ**: 21.62 M€ - Secondary exporter, mainly to PAC

**Net Importers with Near-Zero Rents:**
- **PAC**: 0.11 M€ - Primary importer, pays for imports
- **OCC**: -4.39 M€ - Some inefficient flows

---

## Formula and Methodology

### Congestion Rent Formula

For each directional flow from region i to region j at time t:

```
CR_{i→j}(t) = F_{i→j}(t) × (P_j(t) - P_i(t))
```

Where:
- **F_{i→j}(t)**: Power flow from i to j at time t (MWh)
- **P_j(t)**: Nodal price in region j at time t (€/MWh)
- **P_i(t)**: Nodal price in region i at time t (€/MWh)

### Total Congestion Rent

```
CR_tot = Σ Σ CR_{i→j}(t) × Δt
```

Where:
- **Δt = 0.5 hours** (30-minute time steps)
- Sum is over all region pairs and all time periods

### Interpretation

- **Positive CR**: Power flows from low-price to high-price region (efficient)
- **Negative CR**: Power flows from high-price to low-price region (inefficient or constrained)
- **Magnitude**: Reflects both volume and price differential

---

## Key Insights and Implications

### 1. Transmission Value

The total congestion rent of **170.97 M€/year** represents:
- Economic value of existing transmission capacity
- Justification for transmission investment
- Lower bound on welfare gains from interregional trade

### 2. Market Efficiency

**High efficiency flows:**
- ARA→PAC: 148.90 M€ (large volume, good price arbitrage)
- NAQ→PAC: 22.27 M€

**Potential inefficiencies:**
- OCC flows show negative rents (-4.39 M€ total)
- May indicate operational constraints or loop flows

### 3. Regional Disparities

- **ARA dominance**: Captures 90% of congestion rents
  - Lowest prices (172.31 €/MWh)
  - Large export capacity to high-price regions
  - Strong competitive position

- **PAC price premium**: Highest prices (187.79 €/MWh)
  - +15.48 €/MWh above ARA
  - Heavy reliance on imports
  - Potential for local generation expansion

### 4. Investment Implications

**Transmission expansion priorities:**
1. **ARA↔PAC** (148.82 M€ net rent)
   - Highest congestion rent suggests capacity constraints
   - Additional capacity would enable more arbitrage
   
2. **NAQ↔PAC** (22.46 M€ net rent)
   - Significant value in expanding this link

**Generation expansion priorities:**
- **PAC region**: High prices indicate generation scarcity
- **OCC region**: Negative congestion rents suggest oversupply or constraints

---

## Comparison with Flexibility Analysis

From previous analysis, interregional exchanges (NET) = 23.3 TWh

**Congestion rent efficiency:**
- **Rent per energy traded**: 170.97 M€ / 23.3 TWh = **7.34 €/MWh**
- This represents the average price differential captured through trade

**Highest-value flows:**
- ARA→PAC: 148.90 M€ / 16.73 TWh = **8.90 €/MWh**
- Above average, indicating this is a particularly valuable corridor

---

## Data Files Generated

1. **congestion_rents_detailed.csv**: All 12 directional flows with details
2. **congestion_rents_summary.csv**: Regional totals and system total
3. **congestion_rents_net.csv**: Net rents by bidirectional pairs
4. **congestion_rents_analysis.png**: Comprehensive 6-panel visualization
5. **congestion_rents_summary.png**: Summary bar charts

---

## Recommendations

### Policy Implications

1. **Transmission planning**: Prioritize ARA-PAC and NAQ-PAC corridors
2. **Market design**: Consider how congestion rents are allocated
3. **Generation siting**: Incentivize new capacity in high-price regions (PAC)
4. **Regional cooperation**: Address OCC negative rents through better coordination

### Further Analysis

1. **Temporal analysis**: When do congestion rents peak?
2. **Constraint analysis**: Which transmission limits are most binding?
3. **Counterfactual**: What if transmission capacity were expanded?
4. **Welfare analysis**: Total system welfare vs. congestion rents

---

## Technical Notes

- **Time resolution**: 30-minute intervals (17,520 periods/year)
- **Price data**: Nodal prices from optimization results
- **Flow data**: Actual flows between regions
- **Sign convention**: Positive rent = efficient arbitrage

---

*Report generated from full_year.csv data*  
*Analysis period: Full calendar year*  
*Four regions: ARA, NAQ, OCC, PAC*
