import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
detailed_df = pd.read_csv('congestion_rents_detailed.csv')
summary_df = pd.read_csv('congestion_rents_summary.csv')
net_df = pd.read_csv('congestion_rents_net.csv')

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# 1. Regional congestion rents (bar chart)
ax1 = plt.subplot(2, 3, 1)
regions_data = summary_df[summary_df['Region'] != 'TOTAL'].copy()
colors = ['#2E86AB' if x > 0 else '#E63946' for x in regions_data['Congestion_Rent_M€']]
bars = ax1.bar(regions_data['Region'], regions_data['Congestion_Rent_M€'], 
               color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Congestion Rent (M€)', fontsize=11, fontweight='bold')
ax1.set_title('Congestion Rents by Region\n(Exports Only)', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom' if height > 0 else 'top', 
             fontsize=10, fontweight='bold')

# 2. Pie chart of total congestion rent by region
ax2 = plt.subplot(2, 3, 2)
positive_regions = regions_data[regions_data['Congestion_Rent_M€'] > 0].copy()
wedges, texts, autotexts = ax2.pie(positive_regions['Congestion_Rent_M€'], 
                                     labels=positive_regions['Region'],
                                     autopct='%1.1f%%',
                                     colors=['#2E86AB', '#A23B72', '#F18F01', '#C9ADA7'],
                                     startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)
ax2.set_title('Distribution of Positive\nCongestion Rents', fontsize=12, fontweight='bold')

# 3. Top flows by congestion rent
ax3 = plt.subplot(2, 3, 3)
top_flows = detailed_df.nlargest(8, 'Congestion_Rent_M€')
colors_flows = ['#2E86AB' if x > 0 else '#E63946' for x in top_flows['Congestion_Rent_M€']]
bars = ax3.barh(range(len(top_flows)), top_flows['Congestion_Rent_M€'], 
                color=colors_flows, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(top_flows)))
ax3.set_yticklabels(top_flows['Flow'], fontsize=9)
ax3.set_xlabel('Congestion Rent (M€)', fontsize=11, fontweight='bold')
ax3.set_title('Top 8 Flows by\nCongestion Rent', fontsize=12, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_flows['Congestion_Rent_M€'])):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
             f'{value:.1f}',
             ha='left' if width > 0 else 'right', va='center', 
             fontsize=8, fontweight='bold')

# 4. Net congestion rents by region pair
ax4 = plt.subplot(2, 3, 4)
colors_net = ['#2E86AB' if x > 0 else '#E63946' for x in net_df['Net_CR_M€']]
bars = ax4.bar(range(len(net_df)), net_df['Net_CR_M€'], 
               color=colors_net, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(net_df)))
ax4.set_xticklabels(net_df['Pair'], rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Net Congestion Rent (M€)', fontsize=11, fontweight='bold')
ax4.set_title('Net Congestion Rents\nby Region Pair', fontsize=12, fontweight='bold')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom' if height > 0 else 'top', 
             fontsize=8, fontweight='bold')

# 5. Price differences vs congestion rents (scatter plot)
ax5 = plt.subplot(2, 3, 5)
scatter = ax5.scatter(detailed_df['Avg_Price_Diff_€/MWh'], 
                      detailed_df['Congestion_Rent_M€'],
                      s=detailed_df['Total_Flow_MWh']/1000,  # Size by flow
                      c=detailed_df['Congestion_Rent_M€'],
                      cmap='RdYlBu_r',
                      alpha=0.6,
                      edgecolors='black',
                      linewidth=1)
ax5.set_xlabel('Average Price Difference (€/MWh)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Congestion Rent (M€)', fontsize=11, fontweight='bold')
ax5.set_title('Price Difference vs Congestion Rent\n(bubble size = flow volume)', 
              fontsize=12, fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax5.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax5.grid(alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('CR (M€)', fontsize=9, fontweight='bold')

# 6. Total summary box
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

total_cr = summary_df[summary_df['Region'] == 'TOTAL']['Congestion_Rent_M€'].values[0]
total_positive = regions_data[regions_data['Congestion_Rent_M€'] > 0]['Congestion_Rent_M€'].sum()
total_negative = regions_data[regions_data['Congestion_Rent_M€'] < 0]['Congestion_Rent_M€'].sum()

summary_text = f"""
CONGESTION RENT SUMMARY

Total Congestion Rent:
{total_cr:.2f} M€

By Region (Exports):
  Positive: {total_positive:.2f} M€
  Negative: {total_negative:.2f} M€

Top Flow:
  {detailed_df.iloc[0]['Flow']}
  {detailed_df.iloc[0]['Congestion_Rent_M€']:.2f} M€

Formula:
  CR_i->j(t) = F_i->j(t) × (P_j(t) - P_i(t))
  
Total = Σ CR_i->j(t) × Δt
"""

ax6.text(0.5, 0.5, summary_text, 
         fontsize=11, 
         ha='center', va='center',
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1))

plt.suptitle('Congestion Rent Analysis - Full Year', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('congestion_rents_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'congestion_rents_analysis.png'")

# Create a simplified summary chart
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Regional congestion rents
regions_data_sorted = regions_data.sort_values('Congestion_Rent_M€', ascending=True)
colors = ['#2E86AB' if x > 0 else '#E63946' for x in regions_data_sorted['Congestion_Rent_M€']]
bars = ax1.barh(regions_data_sorted['Region'], regions_data_sorted['Congestion_Rent_M€'], 
                color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_xlabel('Congestion Rent (M€)', fontsize=13, fontweight='bold')
ax1.set_title('Congestion Rents by Region\n(Exports Only)', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for bar, value in zip(bars, regions_data_sorted['Congestion_Rent_M€']):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2.,
             f' {value:.1f} M€',
             ha='left' if width > 0 else 'right', va='center', 
             fontsize=11, fontweight='bold')

# Net congestion rents by pair
net_df_sorted = net_df.sort_values('Net_CR_M€', ascending=True)
colors_net = ['#2E86AB' if x > 0 else '#E63946' for x in net_df_sorted['Net_CR_M€']]
bars = ax2.barh(net_df_sorted['Pair'], net_df_sorted['Net_CR_M€'], 
                color=colors_net, edgecolor='black', linewidth=2, alpha=0.8)
ax2.set_xlabel('Net Congestion Rent (M€)', fontsize=13, fontweight='bold')
ax2.set_title('Net Congestion Rents by Region Pair', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

for bar, value in zip(bars, net_df_sorted['Net_CR_M€']):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
             f' {value:.1f} M€',
             ha='left' if width > 0 else 'right', va='center', 
             fontsize=11, fontweight='bold')

# Add total at bottom
fig2.text(0.5, 0.02, f'Total Congestion Rent: {total_cr:.2f} M€', 
          ha='center', fontsize=13, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('congestion_rents_summary.png', dpi=300, bbox_inches='tight')
print("Summary visualization saved to 'congestion_rents_summary.png'")

plt.show()
