import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Set the visual style
sns.set_theme(style="darkgrid")
# High font scale for readable axis labels in the report
sns.set_context("paper", font_scale=1.4) 

RESULTS_DIR = "results"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Reads all CSVs from results/ and combines them into one DataFrame."""
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    
    df_list = []
    for filename in all_files:
        try:
            basename = os.path.basename(filename).replace(".csv", "")
            parts = basename.split("_")
            seed = int(parts[-1].replace("s", ""))
            
            is_dynamic = "dynamic" in basename
            is_sparse = "sparse" in basename
            
            if "ddqn" in basename:
                algo = "DDQN"
            elif "dqn" in basename:
                algo = "DQN"
            elif "tabq" in basename:
                algo = "TabQ"
            elif "heuristic" in basename:
                algo = "Heuristic"
            else:
                raise ValueError(f"Unknown algorithm: {basename}")

            df = pd.read_csv(filename)
            df["Algorithm"] = algo
            df["Environment"] = "Dynamic" if is_dynamic else "Static"
            df["Reward Type"] = "Sparse" if is_sparse else "Shaped"
            df["Seed"] = seed
            
            if "total_steps" not in df.columns:
                df["total_steps"] = df["steps"].cumsum()
                
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list:
        print("No CSV files found in results/!")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

def plot_metrics(df):
    """Generates plots with smaller, transparent legends."""
    
    # Pre-processing
    df["rolling_success"] = df.groupby(["Algorithm", "Environment", "Reward Type", "Seed"])["success"] \
                              .transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    df["rolling_reward"] = df.groupby(["Algorithm", "Environment", "Reward Type", "Seed"])["reward"].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean())

    df["rolling_time_to_goal"] = df.groupby(["Algorithm", "Environment", "Reward Type", "Seed"])["steps"].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean())
                          
    df_shaped = df[df["Reward Type"] == "Shaped"].copy()
    
    if df_shaped.empty:
        print("Warning: No 'Shaped' reward data found. Using all data for main plots.")
        df_shaped = df

    # --- Plot 1: Average Return ---
    print("Generating Plot 1: Average Return...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_shaped, x="episode", y="rolling_reward", 
        hue="Algorithm", style="Environment", 
        estimator="mean", errorbar=("sd", 1)
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Return (Rolling 30 eps)")
    # framealpha=0.5 makes it 50% transparent
    plt.legend(loc='best', framealpha=0.2, fontsize=12) 
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_average_return.png", bbox_inches="tight")
    plt.close()

    # --- Plot 2: Success Rate ---
    print("Generating Plot 2: Success Rate...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_shaped, x="episode", y="rolling_success", 
        hue="Algorithm", style="Environment"
    )
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (Rolling 30 eps)")
    # framealpha=0.2 makes it 20% transparent
    plt.legend(loc='best', framealpha=0.2, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_success_rate.png", bbox_inches="tight")
    plt.close()

    # --- Plot 3: Time-to-Goal ---
    print("Generating Plot 3: Time-to-Goal...")
    success_df = df_shaped[df_shaped["success"] == 1]
    if not success_df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=success_df, x="episode", y="rolling_time_to_goal", 
            hue="Algorithm", style="Environment"
        )
        plt.xlabel("Episode")
        plt.ylabel("Time-to-Goal (Rolling 30 eps)")
        # framealpha=0.5 makes it 50% transparent
        plt.legend(loc='best', framealpha=0.5, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/3_time_to_goal.png", bbox_inches="tight")
        plt.close()

    # --- Plot 4: Sample Efficiency ---
    print("Generating Plot 4: Sample Efficiency...")
    df_shaped["step_bin"] = (df_shaped["total_steps"] // 1500) * 1500
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df_shaped, x="step_bin", y="reward", 
        hue="Algorithm", style="Environment"
    )
    plt.xlabel("Total Environment Steps")
    plt.ylabel("Average Return") 
    # framealpha=0.2 makes it 20% transparent
    plt.legend(loc='best', framealpha=0.2, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_sample_efficiency.png", bbox_inches="tight")
    plt.close()

    # --- Plot 5: Static vs Dynamic Bar Chart ---
    print("Generating Plot 5: Static vs Dynamic...")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_shaped, x="Algorithm", y="success", 
        hue="Environment"
    )
    plt.xlabel("Algorithm")
    plt.ylabel("Success Rate (Mean)")
    # framealpha=0.2 makes it 20% transparent
    plt.legend(loc='best', framealpha=0.5, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_static_vs_dynamic.png", bbox_inches="tight")
    plt.close()

    # --- Plot 6: Ablation Study (Stacked) ---
    print("Generating Plot 6: Ablation...")
    
    df_rl = df[df["Algorithm"].isin(["TabQ", "DQN", "DDQN"])]
    
    if not df_rl.empty:
        g = sns.relplot(
            data=df_rl, 
            x="episode", y="rolling_success", 
            hue="Reward Type", style="Algorithm",
            row="Environment", kind="line", 
            height=3.5, aspect=2.5
        )
        
        g.set_axis_labels("Episode", "Success Rate (Rolling 30 eps)")
        
        # Legend is outside (top), so we keep frameon=False (no box needed)
        sns.move_legend(
            g, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=4, title=None, frameon=False
        )
        
        g.fig.tight_layout() 
        g.fig.savefig(f"{OUTPUT_DIR}/6_shaping_ablation.png", bbox_inches="tight")
        plt.close()
    
    print(f"\nAll plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    data = load_data()
    if not data.empty:
        plot_metrics(data)