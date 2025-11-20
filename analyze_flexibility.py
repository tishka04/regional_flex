import pandas as pd
import numpy as np

# Load the CSV file
print("Loading data...")
df = pd.read_csv('full_year.csv', sep=';', low_memory=False)

# Convert all numeric columns to float, replacing empty strings and errors with 0
for col in df.columns:
    if col != 'timestamp':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Define the four regions
regions = ['Auvergne_Rhone_Alpes', 'Nouvelle_Aquitaine', 'Occitanie', 'Provence_Alpes_Cote_dAzur']

# 1. Calculate Interregional Exchanges (NET flows to avoid double-counting)
# For each pair of regions, calculate net exchange
region_pairs = [
    ('Auvergne_Rhone_Alpes', 'Nouvelle_Aquitaine'),
    ('Auvergne_Rhone_Alpes', 'Occitanie'),
    ('Auvergne_Rhone_Alpes', 'Provence_Alpes_Cote_dAzur'),
    ('Nouvelle_Aquitaine', 'Occitanie'),
    ('Nouvelle_Aquitaine', 'Provence_Alpes_Cote_dAzur'),
    ('Occitanie', 'Provence_Alpes_Cote_dAzur')
]

interregional_exchanges = 0
net_flow_details = []

for region_a, region_b in region_pairs:
    flow_a_to_b = df[f'flow_out_{region_a}_{region_b}'].sum()
    flow_b_to_a = df[f'flow_out_{region_b}_{region_a}'].sum()
    
    # Net flow from A to B (positive means A exports to B, negative means B exports to A)
    net_flow = flow_a_to_b - flow_b_to_a
    
    # Use absolute value to count total net exchange magnitude
    interregional_exchanges += abs(net_flow)
    
    net_flow_details.append({
        'pair': f'{region_a.split("_")[0]}-{region_b.split("_")[0]}',
        'flow_a_to_b': flow_a_to_b,
        'flow_b_to_a': flow_b_to_a,
        'net_flow': net_flow
    })

print(f"\nRegion pairs analyzed: {len(region_pairs)}")

# 2. Calculate Demand Response + Storage
# Demand response columns
dr_cols = [f'demand_response_{region}' for region in regions]
# Storage discharge columns (both STEP and batteries)
storage_discharge_cols = []
for region in regions:
    storage_discharge_cols.append(f'storage_discharge_STEP_{region}')
    storage_discharge_cols.append(f'storage_discharge_batteries_{region}')
# Storage discharge columns (both STEP and batteries)
storage_charge_cols = []
for region in regions:
    storage_charge_cols.append(f'storage_charge_STEP_{region}')
    storage_charge_cols.append(f'storage_charge_batteries_{region}')

# Sum all demand response and storage discharge
demand_response_total = df[dr_cols].sum().sum()
storage_discharge_total = df[storage_discharge_cols].sum().sum()
storage_charge_total = df[storage_charge_cols].sum().sum()
dr_storage_total = demand_response_total + storage_discharge_total + storage_charge_total

# 3. Calculate Dispatchable Generation
# All dispatch columns for all generation types and regions
dispatch_types = ['hydro', 'nuclear', 'thermal_gas', 'thermal_fuel', 'biofuel']
dispatch_cols = []
for region in regions:
    for gen_type in dispatch_types:
        dispatch_cols.append(f'dispatch_{gen_type}_{region}')

# Sum all dispatchable generation
dispatchable_generation = df[dispatch_cols].sum().sum()

# Calculate total flexible energy
total_flexible = interregional_exchanges + dr_storage_total + dispatchable_generation

# Calculate relative shares
share_exchanges = (interregional_exchanges / total_flexible) * 100
share_dr_storage = (dr_storage_total / total_flexible) * 100
share_dispatchable = (dispatchable_generation / total_flexible) * 100

# Print results
print("\n" + "="*60)
print("FLEXIBLE ENERGY ANALYSIS - FULL YEAR")
print("="*60)
print(f"\n1. Interregional Exchanges (NET flows between regions)")
print(f"   Total Energy: {interregional_exchanges:,.2f} MWh")
print(f"   Relative Share: {share_exchanges:.2f}%")
print(f"   Note: Calculated as net exchanges to avoid double-counting bidirectional flows")

print(f"\n2. Demand Response + Storage")
print(f"   - Demand Response: {demand_response_total:,.2f} MWh")
print(f"   - Storage Discharge: {storage_discharge_total:,.2f} MWh")
print(f"   - Storage Charge: {storage_charge_total:,.2f} MWh")
print(f"   - Total: {dr_storage_total:,.2f} MWh")
print(f"   Relative Share: {share_dr_storage:.2f}%")

print(f"\n3. Dispatchable Generation (Hydro + Nuclear + Thermal + Gas + Biofuel)")
print(f"   Total Energy: {dispatchable_generation:,.2f} MWh")
print(f"   Relative Share: {share_dispatchable:.2f}%")

print(f"\n{'='*60}")
print(f"TOTAL FLEXIBLE ENERGY: {total_flexible:,.2f} MWh")
print(f"TOTAL SHARES: {share_exchanges + share_dr_storage + share_dispatchable:.2f}%")
print("="*60)

# Additional breakdown by generation type
print("\n" + "="*60)
print("DETAILED BREAKDOWN - DISPATCHABLE GENERATION")
print("="*60)
for gen_type in dispatch_types:
    type_cols = [f'dispatch_{gen_type}_{region}' for region in regions]
    type_total = df[type_cols].sum().sum()
    type_share = (type_total / dispatchable_generation) * 100
    print(f"{gen_type.replace('_', ' ').title()}: {type_total:,.2f} MWh ({type_share:.2f}% of dispatchable)")

# Show net flows between region pairs
print("\n" + "="*60)
print("NET INTERREGIONAL EXCHANGES BY REGION PAIR")
print("="*60)
for detail in net_flow_details:
    print(f"\n{detail['pair']}:")
    print(f"  Total A->B: {detail['flow_a_to_b']:,.2f} MWh")
    print(f"  Total B->A: {detail['flow_b_to_a']:,.2f} MWh")
    print(f"  Net Flow:   {detail['net_flow']:,.2f} MWh ({'A->B' if detail['net_flow'] > 0 else 'B->A'})")
    print(f"  |Net|:      {abs(detail['net_flow']):,.2f} MWh")

# Save summary to CSV
summary_data = {
    'Category': ['Interregional Exchanges', 'Demand Response + Storage', 'Dispatchable Generation', 'TOTAL'],
    'Energy (MWh)': [interregional_exchanges, dr_storage_total, dispatchable_generation, total_flexible],
    'Share (%)': [share_exchanges, share_dr_storage, share_dispatchable, 100.0]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('flexibility_shares_summary.csv', index=False)
print("\nSummary saved to 'flexibility_shares_summary.csv'")
