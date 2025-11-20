import pandas as pd
import numpy as np

# Load the CSV file
print("Loading data...")
df = pd.read_csv('full_year.csv', sep=';', low_memory=False)

# Convert all numeric columns to float
for col in df.columns:
    if col != 'timestamp':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Define regions
regions = ['Auvergne_Rhone_Alpes', 'Nouvelle_Aquitaine', 'Occitanie', 'Provence_Alpes_Cote_dAzur']
region_short = {
    'Auvergne_Rhone_Alpes': 'ARA',
    'Nouvelle_Aquitaine': 'NAQ',
    'Occitanie': 'OCC',
    'Provence_Alpes_Cote_dAzur': 'PAC'
}

# Time step duration (30-minute intervals = 0.5 hours)
delta_t = 0.5

# Get all flow and price columns
flow_cols = [col for col in df.columns if 'flow_out' in col]
price_cols = {region: f'nodal_price_{region}' for region in regions}

print(f"\nFound {len(flow_cols)} flow columns")
print(f"Found {len(price_cols)} price columns")

# Calculate congestion rents for each directional flow
congestion_rents = {}
detailed_results = []

for flow_col in flow_cols:
    # Parse the flow column name to extract source and destination regions
    # Format: flow_out_SourceRegion_DestinationRegion
    parts = flow_col.replace('flow_out_', '').split('_')
    
    # Reconstruct region names (they may have underscores)
    # We need to find where one region ends and the next begins
    found = False
    for region_i in regions:
        if flow_col.startswith(f'flow_out_{region_i}_'):
            source = region_i
            # The destination is what remains after removing source
            dest_part = flow_col.replace(f'flow_out_{region_i}_', '')
            for region_j in regions:
                if dest_part == region_j:
                    destination = region_j
                    found = True
                    break
            if found:
                break
    
    if not found:
        print(f"Warning: Could not parse flow column: {flow_col}")
        continue
    
    # Get flow and prices
    F_ij = df[flow_col]  # Flow from i to j
    P_i = df[price_cols[source]]  # Price in source region
    P_j = df[price_cols[destination]]  # Price in destination region
    
    # Calculate congestion rent at each timestep: CR_ij(t) = F_ij(t) * (P_j(t) - P_i(t))
    CR_ij_t = F_ij * (P_j - P_i)
    
    # Total congestion rent for this flow: sum over all timesteps * delta_t
    CR_total = (CR_ij_t * delta_t).sum()
    
    # Store results
    flow_name = f"{region_short[source]}->{region_short[destination]}"
    congestion_rents[flow_name] = CR_total
    
    detailed_results.append({
        'Flow': flow_name,
        'Source': region_short[source],
        'Destination': region_short[destination],
        'Total_Flow_MWh': F_ij.sum(),
        'Avg_Price_Source_€/MWh': P_i.mean(),
        'Avg_Price_Dest_€/MWh': P_j.mean(),
        'Avg_Price_Diff_€/MWh': (P_j - P_i).mean(),
        'Congestion_Rent_€': CR_total,
        'Congestion_Rent_M€': CR_total / 1e6
    })

# Calculate total congestion rent
CR_total_all = sum(congestion_rents.values())

# Calculate congestion rents by region (sum of all flows originating from that region)
region_congestion_rents = {}
for region in regions:
    region_cr = 0
    for flow_name, cr_value in congestion_rents.items():
        if flow_name.startswith(region_short[region] + '->'):
            region_cr += cr_value
    region_congestion_rents[region_short[region]] = region_cr

# Print results
print("\n" + "="*70)
print("CONGESTION RENT ANALYSIS - FULL YEAR")
print("="*70)
print(f"\nFormula: CR_i->j(t) = F_i->j(t) * (P_j(t) - P_i(t))")
print(f"Time step: Delta_t = {delta_t} hours")

print("\n" + "="*70)
print("CONGESTION RENTS BY DIRECTIONAL FLOW")
print("="*70)

# Sort by absolute congestion rent
sorted_flows = sorted(detailed_results, key=lambda x: abs(x['Congestion_Rent_€']), reverse=True)

for result in sorted_flows:
    print(f"\n{result['Flow']}:")
    print(f"  Total Flow:           {result['Total_Flow_MWh']:>15,.2f} MWh")
    print(f"  Avg Price Source:     {result['Avg_Price_Source_€/MWh']:>15,.2f} €/MWh")
    print(f"  Avg Price Dest:       {result['Avg_Price_Dest_€/MWh']:>15,.2f} €/MWh")
    print(f"  Avg Price Difference: {result['Avg_Price_Diff_€/MWh']:>15,.2f} €/MWh")
    print(f"  Congestion Rent:      {result['Congestion_Rent_M€']:>15,.2f} M€")

print("\n" + "="*70)
print("CONGESTION RENTS BY REGION (Exports Only)")
print("="*70)
for region_code, cr_value in sorted(region_congestion_rents.items(), 
                                     key=lambda x: abs(x[1]), reverse=True):
    print(f"{region_code}: {cr_value/1e6:>15,.2f} M€")

print("\n" + "="*70)
print(f"TOTAL CONGESTION RENT (ALL FLOWS): {CR_total_all/1e6:>10,.2f} M€")
print(f"                                   {CR_total_all:>15,.2f} €")
print("="*70)

# Save detailed results to CSV
results_df = pd.DataFrame(detailed_results)
results_df = results_df.sort_values('Congestion_Rent_€', key=abs, ascending=False)
results_df.to_csv('congestion_rents_detailed.csv', index=False)
print("\nDetailed results saved to 'congestion_rents_detailed.csv'")

# Create summary table
summary_data = []
for region_code, cr_value in region_congestion_rents.items():
    summary_data.append({
        'Region': region_code,
        'Congestion_Rent_€': cr_value,
        'Congestion_Rent_M€': cr_value / 1e6
    })

summary_data.append({
    'Region': 'TOTAL',
    'Congestion_Rent_€': CR_total_all,
    'Congestion_Rent_M€': CR_total_all / 1e6
})

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('congestion_rents_summary.csv', index=False)
print("Summary saved to 'congestion_rents_summary.csv'")

# Calculate net congestion rents (considering bidirectional flows)
print("\n" + "="*70)
print("NET CONGESTION RENTS (Bidirectional Pairs)")
print("="*70)

region_pairs = [
    ('ARA', 'NAQ'),
    ('ARA', 'OCC'),
    ('ARA', 'PAC'),
    ('NAQ', 'OCC'),
    ('NAQ', 'PAC'),
    ('OCC', 'PAC')
]

net_congestion_rents = []
for region_a, region_b in region_pairs:
    flow_ab = f"{region_a}->{region_b}"
    flow_ba = f"{region_b}->{region_a}"
    
    cr_ab = congestion_rents.get(flow_ab, 0)
    cr_ba = congestion_rents.get(flow_ba, 0)
    
    net_cr = cr_ab + cr_ba
    
    print(f"\n{region_a} <-> {region_b}:")
    print(f"  CR {flow_ab}: {cr_ab/1e6:>10,.2f} M€")
    print(f"  CR {flow_ba}: {cr_ba/1e6:>10,.2f} M€")
    print(f"  Net CR:       {net_cr/1e6:>10,.2f} M€")
    
    net_congestion_rents.append({
        'Pair': f"{region_a}<->{region_b}",
        'CR_A_to_B_M€': cr_ab/1e6,
        'CR_B_to_A_M€': cr_ba/1e6,
        'Net_CR_M€': net_cr/1e6
    })

net_cr_df = pd.DataFrame(net_congestion_rents)
net_cr_df.to_csv('congestion_rents_net.csv', index=False)
print("\n\nNet congestion rents saved to 'congestion_rents_net.csv'")
