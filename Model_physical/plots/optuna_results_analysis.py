import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import optuna
from optuna.importance import get_param_importances

# Set global scientific style
plt.style.use(['science', 'ieee'])

# Paths
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
DB_PATH = os.path.join(BASE_DIR, "Model_physical", "Results", "V7_model_optimized", "optuna_study.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "plots", "output")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "all_countries_best_params.csv")

def get_best_params_and_importance():
    storage_url = f"sqlite:///{DB_PATH}"
    study_summaries = optuna.get_all_study_summaries(storage=storage_url)
    
    all_best_params = []
    importance_data = []
    
    print(f"Loading {len(study_summaries)} studies from {DB_PATH}...")
    
    for summary in study_summaries:
        study_name = summary.study_name
        # Pattern: V7_Optimization_Country_Name_correlation
        if not study_name.startswith("V7_Optimization_") or not study_name.endswith("_correlation"):
            continue
            
        country = study_name.replace("V7_Optimization_", "").replace("_correlation", "").replace("_", " ")
        
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # Best Params
        if study.best_trial:
            params = study.best_params.copy()
            params['Country'] = country
            params['Best_Score'] = study.best_value
            all_best_params.append(params)
            
            # Hyperparameter Importance
            try:
                importances = get_param_importances(study)
                importances['Country'] = country
                importance_data.append(importances)
            except Exception as e:
                print(f"Could not calculate importance for {country}: {e}")
                
    df_params = pd.DataFrame(all_best_params)
    
    # --- Formatting Rules ---
    if not df_params.empty:
        # Reorder columns: Country first
        cols = ['Country'] + [c for c in df_params.columns if c != 'Country']
        df_params = df_params[cols]

        # T_OPT and Critical Window days: integer
        if 'T_OPT' in df_params.columns:
            df_params['T_OPT'] = df_params['T_OPT'].astype(int)
        if 'CRITICAL_WINDOW_DAYS' in df_params.columns:
            df_params['CRITICAL_WINDOW_DAYS'] = df_params['CRITICAL_WINDOW_DAYS'].astype(int)
        
        # Best_Score (correlation): two numbers after comma
        if 'Best_Score' in df_params.columns:
            df_params['Best_Score'] = df_params['Best_Score'].round(2)
            
        # All other numbers 1 number after the comma
        other_cols = [c for c in df_params.columns if c not in ['Country', 'Best_Score', 'T_OPT', 'CRITICAL_WINDOW_DAYS']]
        for col in other_cols:
            if df_params[col].dtype in [np.float64, np.float32]:
                df_params[col] = df_params[col].round(1)
    
    df_importance = pd.DataFrame(importance_data)
    
    return df_params, df_importance

def plot_parameter_distributions(df_params):
    print("Generating parameter distributions plot...")
    # Identify hyperparameter columns (exclude Country and Best_Score)
    param_cols = [c for c in df_params.columns if c not in ['Country', 'Best_Score']]
    n_params = len(param_cols)
    
    width = 7.0
    height = width * 0.8 # Slightly taller to accommodate multiple subplots
    
    fig, axes = plt.subplots(int(np.ceil(n_params/2)), 2, figsize=(width, height))
    axes = axes.flatten()
    
    for i, col in enumerate(param_cols):
        ax = axes[i]
        # Using a violin plot + boxplot for "formal" distribution view
        parts = ax.violinplot(df_params[col], showmeans=False, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#4c72b0')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        ax.set_title(col.replace("_", " "), fontsize=10)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
        
        # Remove x-ticks for single violin
        ax.set_xticks([])
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "parameter_distributions.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "parameter_distributions.png"), format='png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_average_importance(df_importance):
    print("Generating average hyperparameter importance plot...")
    # Exclude Country column
    imp_cols = [c for c in df_importance.columns if c != 'Country']
    avg_importance = df_importance[imp_cols].mean().sort_values(ascending=True)
    
    width = 3.3 # Single column width often looks better for 1-pane horizontal bar charts
    height = width * 0.8
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    bars = ax.barh(avg_importance.index, avg_importance.values, color='#4c72b0', edgecolor='black', linewidth=0.5)
    
    # Add values at the end of bars
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val + 0.01, bar.get_y() + bar.get_height()/2, f'{width_val:.2f}', 
                va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel("Mean Hyperparameter Importance", fontsize=9)
    ax.set_title("Global Hyperparameter Impact", fontsize=10)
    ax.set_xlim(0, avg_importance.max() * 1.25) # Add space for labels
    
    # Clean up labels
    ax.set_yticklabels([idx.replace("_", " ") for idx in avg_importance.index], fontsize=8)
    
    ax.grid(True, axis='x', linestyle='--', linewidth=0.3, alpha=0.5)
    
    # Stylize spines (like in fig_correlation_with_fao_V7.py)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_param_importance.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "average_param_importance.png"), format='png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_params, df_importance = get_best_params_and_importance()
    
    if df_params.empty:
        print("No results found in the database.")
        return
        
    # 1. Save CSV
    df_params.to_csv(CSV_OUTPUT, index=False)
    print(f"Saved aggregated params to {CSV_OUTPUT}")
    
    # 2. Plot Distributions
    plot_parameter_distributions(df_params)
    
    # 3. Plot Importance
    if not df_importance.empty:
        plot_average_importance(df_importance)
    else:
        print("No importance data available to plot.")

    print("\nAnalysis complete. Files generated in:")
    print(f" - {CSV_OUTPUT}")
    print(f" - {os.path.join(OUTPUT_DIR, 'parameter_distributions.pdf/png')}")
    print(f" - {os.path.join(OUTPUT_DIR, 'average_param_importance.pdf/png')}")

if __name__ == "__main__":
    main()
