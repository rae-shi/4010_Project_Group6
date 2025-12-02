import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Set the visual style
sns.set_theme(style="darkgrid")
RESULTS_DIR = "results"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Reads all CSVs from results/ and combines them into one DataFrame."""
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    
    df_list = []
    for filename in all_files:
        try:
            # Filename format: algo_dynamic_sparse_s20.csv
            basename = os.path.basename(filename).replace(".csv", "")
            parts = basename.split("_")
            
            # Extract Seed
            seed = int(parts[-1].replace("s", ""))
            
            # Extract Settings
            is_dynamic = "dynamic" in basename
            is_sparse = "sparse" in basename
            
            # Extract Algo Name
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

            # Load CSV
            df = pd.read_csv(filename)
            
            # Add metadata columns
            df["Algorithm"] = algo
            df["Environment"] = "Dynamic" if is_dynamic else "Static"
            df["Reward Type"] = "Sparse" if is_sparse else "Shaped"
            df["Seed"] = seed
            
            # Calculate cumulative steps if missing
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
    """Generates the requested plots."""
    
    # Pre-processing
    # Calculate rolling success rate for everything first
    df["rolling_success"] = df.groupby(["Algorithm", "Environment", "Reward Type", "Seed"])["success"] \
                              .transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    df["rolling_reward"] = df.groupby(["Algorithm", "Environment", "Reward Type", "Seed"])["reward"].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean())

    df["rolling_time_to_goal"] = df.groupby(["Algorithm", "Environment", "Reward Type", "Seed"])["steps"].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)                          
    df_shaped = df[df["Reward Type"] == "Shaped"].copy()
    
    if df_shaped.empty:
        print("Warning: No 'Shaped' reward data found. Using all data for main plots.")
        df_shaped = df

    # Plots 1-5: Main Performance (Shaped Only)
    
    # 1. Average Return
    print("Generating Plot 1: Average Return (Shaped)...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_shaped, x="episode", y="rolling_reward", 
        hue="Algorithm", style="Environment", 
        estimator="mean", errorbar=("sd", 1)
    )
    plt.ylabel("Average Return (Rolling 30 eps)")
    plt.title("Average Return per Episode (Shaped Reward)")
    plt.savefig(f"{OUTPUT_DIR}/1_average_return.png")
    plt.close()

    # 2. Success Rate
    print("Generating Plot 2: Success Rate (Shaped)...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_shaped, x="episode", y="rolling_success", 
        hue="Algorithm", style="Environment"
    )
    plt.ylabel("Success Rate (Rolling 30 eps)")
    plt.title("Success Rate vs Episode (Shaped Reward)")
    plt.savefig(f"{OUTPUT_DIR}/2_success_rate.png")
    plt.close()

    # 3. Time-to-Goal
    print("Generating Plot 3: Time-to-Goal (Shaped)...")
    success_df = df_shaped[df_shaped["success"] == 1]
    if not success_df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=success_df, x="episode", y="rolling_time_to_goal", 
            hue="Algorithm", style="Environment"
        )
        plt.ylabel("Time-to-Goal (Rolling 30 eps)")
        plt.title("Time-to-Goal (Successful Episodes)")
        plt.savefig(f"{OUTPUT_DIR}/3_time_to_goal.png")
        plt.close()

    # 4. Sample Efficiency
    print("Generating Plot 4: Sample Efficiency (Shaped)...")
    df_shaped["step_bin"] = (df_shaped["total_steps"] // 1500) * 1500
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_shaped, x="step_bin", y="reward", 
        hue="Algorithm", style="Environment"
    )
    plt.title("Sample Efficiency (Shaped Reward)")
    plt.savefig(f"{OUTPUT_DIR}/4_sample_efficiency.png")
    plt.close()

    # 5. Static vs Dynamic Bar Chart
    print("Generating Plot 5: Static vs Dynamic (Shaped)...")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_shaped, x="Algorithm", y="success", 
        hue="Environment"
    )
    plt.title("Overall Success Rate: Static vs Dynamic")
    plt.savefig(f"{OUTPUT_DIR}/5_static_vs_dynamic.png")
    plt.close()

    # Plot 6: ABLATION STUDY (Sparse vs Shaped)
    print("Generating Plot 6: Reward Shaping Comparison...")
    
    # Filter for only RL agents (Heuristic doesn't learn, so shaping doesn't matter)
    df_rl = df[df["Algorithm"].isin(["TabQ", "DQN", "DDQN"])]
    
    if not df_rl.empty:
        # Create a FacetGrid: Columns = Environment, Hue = Reward Type
        g = sns.relplot(
            data=df_rl, 
            x="episode", y="rolling_success", 
            hue="Reward Type", style="Algorithm",
            col="Environment", kind="line",
            height=5, aspect=1.2
        )
        g.fig.suptitle("Impact of Reward Shaping: Sparse vs Shaped", y=1.02)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95]) 
        g.fig.savefig(f"{OUTPUT_DIR}/6_shaping_ablation.png", bbox_inches="tight")
        plt.close()
    
    print(f"\nAll plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    data = load_data()
    if not data.empty:
        plot_metrics(data)