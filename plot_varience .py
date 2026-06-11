

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_benchmark_plot():
    csv_path = '../build/final_weights.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Run the C++ executable first!")
        return

    # 1. Load the C++ benchmark results
    df = pd.read_csv(csv_path)
    
    # 2. Inject the True Market Reality
    df['True_Reality'] = [3.0, 0.0]
    
    # 3. Reshape for side-by-side plotting
    df_melted = df.melt(id_vars='Feature', var_name='Source', value_name='Weight')
    
    # 4. Generate the graph
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_melted, x='Feature', y='Weight', hue='Source', 
                palette={'Learned_Weight': '#3498db', 'True_Reality': '#2ecc71'},
                edgecolor='black')
    
    plt.title('RSW Covariate-Shift Elimination', fontsize=16, fontweight='bold')
    plt.suptitle('Comparing the C++ RSW engine against true invariant market mechanics', fontsize=11, color='gray')
    plt.ylabel('Coefficient Weight', fontsize=12)
    plt.xlabel('Feature Type', fontsize=12)
    plt.axhline(0, color='black', linewidth=1.2)
    
    # 5. Export the graph
    output_path = '../build/rsw_benchmark_results.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Success! Graph saved to {output_path}")

if __name__ == "__main__":
    generate_benchmark_plot()