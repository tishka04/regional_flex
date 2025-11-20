import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the summary data
df = pd.read_csv('flexibility_shares_summary.csv')

# Remove the TOTAL row for visualization
df_plot = df[df['Category'] != 'TOTAL'].copy()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Color scheme
colors = ['#2E86AB', '#A23B72', '#F18F01']

# 1. Bar chart of shares
bars = ax1.bar(range(len(df_plot)), df_plot['Share (%)'], color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
ax1.set_title('Relative Shares of Flexible Energy\n(Full Year)', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(df_plot)))
ax1.set_xticklabels(df_plot['Category'], rotation=15, ha='right', fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 90)

# Add percentage labels on bars
for i, (bar, value) in enumerate(zip(bars, df_plot['Share (%)'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. Pie chart
wedges, texts, autotexts = ax2.pie(df_plot['Share (%)'], 
                                     labels=df_plot['Category'],
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90,
                                     textprops={'fontsize': 10})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax2.set_title('Distribution of Flexible Energy Sources', fontsize=14, fontweight='bold')

# Add summary text box
textstr = f'Total Flexible Energy: {df[df["Category"]=="TOTAL"]["Energy (MWh)"].values[0]:,.0f} MWh'
fig.text(0.5, 0.02, textstr, ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('flexibility_shares.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'flexibility_shares.png'")

# Create a detailed breakdown visualization
fig2, ax = plt.subplots(figsize=(12, 7))

# Data for visualization
categories = ['Interregional\nExchanges', 'DR + Storage', 'Dispatchable\nGeneration']
shares = df_plot['Share (%)'].values
energies = df_plot['Energy (MWh)'].values

# Create bars
x = np.arange(len(categories))
bars = ax.bar(x, shares, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

# Customize
ax.set_ylabel('Share of Total Flexible Energy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Flexibility Source', fontsize=13, fontweight='bold')
ax.set_title('Flexible Energy Provision - Annual Analysis\n', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
ax.set_ylim(0, 90)

# Add dual labels (percentage and energy)
for i, (bar, share, energy) in enumerate(zip(bars, shares, energies)):
    height = bar.get_height()
    # Percentage
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{share:.2f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')
    # Energy value
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{energy/1e6:.1f}\nTWh',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Add reference line
ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% threshold')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('flexibility_analysis_detailed.png', dpi=300, bbox_inches='tight')
print("Detailed visualization saved to 'flexibility_analysis_detailed.png'")

plt.show()
