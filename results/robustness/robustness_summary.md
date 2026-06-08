# Robustness Analysis Summary

## Hydro Sensitivity

| scenario_label | total_cost_meur | hydro_share_pct | nuclear_share_pct | gas_share_pct | dr_mwh | storage_throughput_mwh | exchanges_mwh | congestion_rents_meur | irre_up_pct_ref | irre_dn_pct_ref | efs_up_mw_ref | efs_dn_mw_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline 2023 | 8437.46 | 11.26 | 88.56 | 0.08 | 0.00 | 57060.82 | 30623907.94 | 905.44 | 28.87 | 32.67 | 316.34 | 279.72 |
| Simplified hydro | 5767.43 | 71.01 | 28.99 | 0.00 | 0.00 | 0.00 | 4144077.17 | 116.56 | 28.87 | 32.67 | 316.34 | 279.72 |

## DR Sensitivity

| scenario_label | recovery_beta | total_cost_meur | dr_mwh | storage_throughput_mwh | exchanges_mwh | congestion_rents_meur |
| --- | --- | --- | --- | --- | --- | --- |
| beta = 0 | 0.00 | 8437.48 | 2.78 | 57061.59 | 30624566.97 | 905.30 |
| beta = 0.5 | 0.50 | 8437.46 | 0.00 | 57060.82 | 30623907.94 | 905.44 |
| beta = 1 | 1.00 | 8437.46 | 0.00 | 57060.82 | 30623907.94 | 905.44 |

## Interannual Robustness

| scenario_label | total_cost_meur | hydro_share_pct | nuclear_share_pct | gas_share_pct | dr_mwh | storage_throughput_mwh | exchanges_mwh | congestion_rents_meur | runtime_total_s | irre_up_pct_ref | irre_dn_pct_ref | efs_up_mw_ref | efs_dn_mw_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 9714.13 | 8.09 | 91.38 | 0.31 | 0.00 | 78438.49 | 35541026.04 | 1084.44 | 518.41 | 28.87 | 32.67 | 316.34 | 279.72 |
| 2023 | 8437.46 | 11.26 | 88.56 | 0.08 | 0.00 | 57060.82 | 30623907.94 | 905.44 | 499.13 | 28.87 | 32.67 | 316.34 | 279.72 |

Figures:
- C:\Users\coudr\projects\regional_flex\plots\robustness\hydro_sensitivity.png
- C:\Users\coudr\projects\regional_flex\plots\robustness\dr_sensitivity.png
