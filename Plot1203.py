# -*- coding: utf-8 -*-
"""
Project: Trajectory of Hesitation - Advanced Analysis
Changes:
1. All text converted to English for publication standards.
2. Direct plotting (plt.show) instead of saving.
3. Added 3 new analytical plots (Psychometric, RT Dist, MD-RT Correlation).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Plotting Style Settings (Publication Quality) ---
sns.set_theme(style="ticks", context="talk", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # Standard English fonts

# Color Palettes
colors_pressure = {'no_pressure': '#2E86C1', 'high_pressure': '#E74C3c'}  # Blue vs Red
colors_sign = {'gain': '#F1C40F', 'loss': '#8E44AD'}  # Gold vs Purple


# ==========================================
# 1. Data Loading & Mock Data Generator
# ==========================================
def load_data():
    # Try to load real data
    path = 'data'
    all_files = glob.glob(os.path.join(path, "*.csv"))

    if len(all_files) > 0:
        print(f"Loading {len(all_files)} files from '{path}'...")
        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                df_list.append(df)
            except:
                pass
        if df_list:
            full_df = pd.concat(df_list, axis=0, ignore_index=True)
            # Basic cleaning
            clean_df = full_df[full_df['rt'] > 0.1].copy()
            # Handle boolean/string conversion
            if clean_df['chose_delayed'].dtype == object:
                clean_df['chose_delayed_int'] = clean_df['chose_delayed'].astype(str).map(
                    {'True': 1, 'False': 0, '1': 1, '0': 0})
            else:
                clean_df['chose_delayed_int'] = clean_df['chose_delayed'].astype(int)
            return clean_df

    print("No data found or load failed. Generating MOCK DATA for demonstration...")
    return generate_mock_data()


def generate_mock_data(n_subs=20):
    # Generates high-fidelity mock data to demonstrate the plotting capabilities
    np.random.seed(42)
    data = []
    for sub in range(1, n_subs + 1):
        for sign in ['gain', 'loss']:
            for pressure in ['no_pressure', 'high_pressure']:
                # True Indifference Point (IP) simulation
                # Pressure makes Gain IP drop (impulsive), Loss IP stable or rise
                true_ip = 85 if sign == 'gain' and pressure == 'no_pressure' else \
                    60 if sign == 'gain' and pressure == 'high_pressure' else \
                        75  # Loss
                true_ip += np.random.normal(0, 5)  # Individual difference

                curr_m = 50
                for t in range(12):  # 12 trials per block
                    # Value difference: Standard (Delay) - Immediate
                    sv_std = true_ip
                    val_diff = (sv_std - curr_m) if sign == 'gain' else (curr_m - sv_std)  # Adjust for loss direction

                    # Sigmoid choice probability
                    sensitivity = 0.15 if pressure == 'no_pressure' else 0.08  # Pressure blunts sensitivity
                    prob_delay = 1 / (1 + np.exp(-val_diff * sensitivity))
                    choice = 1 if np.random.rand() < prob_delay else 0

                    # RT simulation (DDM style)
                    difficulty = abs(val_diff)
                    base_rt = 0.8 if pressure == 'no_pressure' else 0.4  # Collapsed bound
                    rt = base_rt + 1.0 * np.exp(-difficulty / 15) + np.random.gamma(2, 0.1)
                    if pressure == 'high_pressure': rt = min(rt, 1.8)

                    # MD simulation (Motor conflict)
                    # Harder trials (low diff) -> higher MD
                    base_md = 0.1
                    md = base_md + 0.3 * np.exp(-difficulty / 10) + np.random.normal(0, 0.02)
                    if sign == 'loss': md += 0.1  # Loss creates more conflict

                    # Trajectory simulation (Simple curve generation)
                    # Create a simple arc based on MD
                    xs = np.linspace(0, 0.4, 20)
                    ys = np.linspace(0, 0.3, 20)
                    # Add curvature
                    ys += np.sin(np.linspace(0, np.pi, 20)) * md
                    traj_str = ";".join([f"{x:.3f},{y:.3f}" for x, y in zip(xs, ys)])

                    data.append({
                        'participant': sub, 'block_sign': sign, 'block_pressure': pressure,
                        'comp_m': curr_m, 'rt': rt, 'chose_delayed_int': choice,
                        'max_deviation': md, 'raw_trajectory': traj_str
                    })

                    # PEST adjustment
                    step = 20 / (t + 1) ** 0.5
                    if sign == 'gain':
                        curr_m += step if choice else -step
                    else:
                        curr_m -= step if choice else -step

    return pd.DataFrame(data)


# ==========================================
# 2. Main Plotting Logic
# ==========================================
def run_plotting():
    df = load_data()

    # --- Pre-calculation ---
    # 1. Calculate IP (Indifference Point)
    df_ip = df.groupby(['participant', 'block_sign', 'block_pressure']).tail(3)
    df_ip_agg = df_ip.groupby(['participant', 'block_sign', 'block_pressure'])['comp_m'].mean().reset_index()

    # 2. Calculate Difficulty & Bins
    df_merged = df.merge(df_ip_agg, on=['participant', 'block_sign', 'block_pressure'], suffixes=('', '_final'))
    df_merged['difficulty'] = abs(df_merged['comp_m'] - df_merged['comp_m_final'])
    df_merged['diff_bin'] = pd.cut(df_merged['difficulty'], bins=[-1, 5, 20, 100], labels=['Hard', 'Medium', 'Easy'])

    # 3. Calculate Real Value Difference (Delay - Immediate) for Psychometric Curve
    # Estimate: Value Diff = IP_final - Current_Immediate
    df_merged['val_diff_real'] = df_merged['comp_m_final'] - df_merged['comp_m']

    print(">>> Generating Plots...")

    # =======================================================
    # Figure 1: Indifference Points (Behavioral Outcome)
    # =======================================================
    plt.figure(figsize=(8, 6))

    sns.barplot(x='block_sign', y='comp_m', hue='block_pressure', data=df_ip_agg,
                palette=colors_pressure, capsize=0.1, alpha=0.8, edgecolor=".2")

    sns.stripplot(x='block_sign', y='comp_m', hue='block_pressure', data=df_ip_agg,
                  dodge=True, color='black', alpha=0.5, size=5, legend=False)

    plt.title("Fig 1. Effect of Pressure on Indifference Points", fontweight='bold', pad=15)
    plt.ylabel("Indifference Point (Amount)")
    plt.xlabel("Context (Sign)")
    plt.ylim(0, 110)
    plt.legend(title='Time Pressure', loc='upper right', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # =======================================================
    # Figure 2: Chronometric Function (DDM Mechanism)
    # =======================================================
    plt.figure(figsize=(9, 6))

    sns.lineplot(x='diff_bin', y='rt', hue='block_pressure', style='block_sign', data=df_merged,
                 palette=colors_pressure, markers=True, linewidth=3, ms=10, err_style='bars')

    plt.title("Fig 2. Chronometric Function (RT vs Difficulty)", fontweight='bold', pad=15)
    plt.ylabel("Reaction Time (s)")
    plt.xlabel("Decision Difficulty")
    plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.3, 1))

    # Annotation
    plt.annotate("Collapsed Bound", xy=(0, 0.5), xytext=(0.5, 1.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5))

    sns.despine()
    plt.tight_layout()
    plt.show()

    # =======================================================
    # Figure 3: Trajectory Max Deviation (Motor Conflict)
    # =======================================================
    plt.figure(figsize=(8, 6))

    # Filter outliers
    df_md = df_merged[df_merged['max_deviation'] < 0.6]

    sns.violinplot(x='block_sign', y='max_deviation', hue='block_pressure', data=df_md,
                   split=True, inner='quartile', palette=colors_pressure, cut=0)

    plt.title("Fig 3. Motor Conflict: Max Deviation", fontweight='bold', pad=15)
    plt.ylabel("Trajectory Curvature (MD)")
    plt.xlabel("Context")
    plt.legend(title='Pressure', loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # =======================================================
    # [NEW] Figure 4: Psychometric Curve (Sensitivity Analysis)
    # 意义：展示选择概率如何随价值差变化。高压组斜率更平，说明敏感度下降。
    # =======================================================
    plt.figure(figsize=(10, 6))

    # Using lmplot logic manually for better control
    for press in ['no_pressure', 'high_pressure']:
        subset = df_merged[df_merged['block_pressure'] == press]
        sns.regplot(x='val_diff_real', y='chose_delayed_int', data=subset,
                    logistic=True, n_boot=100, ci=95,
                    scatter_kws={'alpha': 0.05}, line_kws={'label': press, 'color': colors_pressure[press]})

    plt.title("Fig 4. Psychometric Curve (Value Sensitivity)", fontweight='bold', pad=15)
    plt.xlabel("Value Difference (Estimated SV - Immediate)")
    plt.ylabel("P(Choose Delay)")
    plt.axvline(0, color='grey', linestyle='--', alpha=0.3)
    plt.axhline(0.5, color='grey', linestyle='--', alpha=0.3)
    plt.legend(title='Pressure')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # =======================================================
    # [NEW] Figure 5: RT Distribution (Collapsed Bound Evidence)
    # 意义：直观展示高压如何把RT分布“挤压”到左侧。
    # =======================================================
    plt.figure(figsize=(9, 6))

    sns.kdeplot(data=df_merged, x='rt', hue='block_pressure', fill=True, common_norm=False,
                palette=colors_pressure, alpha=0.3, linewidth=2)

    plt.axvline(1.5, color='red', linestyle='--', alpha=0.5, label='Time Limit (1.5s)')

    plt.title("Fig 5. Reaction Time Distributions", fontweight='bold', pad=15)
    plt.xlabel("Reaction Time (s)")
    plt.ylabel("Density")
    plt.xlim(0, 3.0)
    plt.legend(title='Pressure')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # =======================================================
    # [NEW] Figure 6: Motor-Decision Correlation (MD vs RT)
    # 意义：验证“思维泄露”。越难的决策(RT长)，手抖得越厉害(MD大)。
    # =======================================================
    g = sns.jointplot(data=df_merged, x="rt", y="max_deviation", hue="block_pressure",
                      palette=colors_pressure, alpha=0.4, height=8)

    g.fig.suptitle("Fig 6. Correlation: Decision Time vs. Motor Conflict", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Reaction Time (s)", "Max Deviation (Curvature)")
    plt.show()

    # =======================================================
    # Figure 7: Trajectory Visualization (Spaghetti Plot)
    # =======================================================
    fig, axes = plt.subplots(1, 2, figsize=(300, 6))

    plot_traj_on_ax(df_merged, 'no_pressure', axes[0], "Fig 7A. Trajectories: No Pressure")
    plot_traj_on_ax(df_merged, 'high_pressure', axes[1], "Fig 7B. Trajectories: High Pressure")

    # tight_layout 自动调整，可指定参数
    plt.tight_layout(
        pad=2.0,  # 图形边缘的填充
        w_pad=4.0,  # 子图之间的水平间距
        h_pad=3.0,  # 子图之间的垂直间距
    )
    plt.show()


# --- Helper for Trajectory Plotting ---
def plot_traj_on_ax(df, press_cond, ax, title):
    sub_df = df[df['block_pressure'] == press_cond]
    count = 0
    for idx, row in sub_df.iterrows():
        try:
            points = str(row['raw_trajectory']).split(';')
            xs = [float(p.split(',')[0]) for p in points]
            ys = [float(p.split(',')[1]) for p in points]
        except:
            continue

        if len(xs) < 2: continue

        # Normalize start to (0,0)
        xs = [x - xs[0] for x in xs]
        ys = [y - ys[0] for y in ys]

        # Flip left choices to right for comparison
        if xs[-1] < 0: xs = [-x for x in xs]

        color = colors_sign['gain'] if row['block_sign'] == 'gain' else colors_sign['loss']
        ax.plot(xs, ys, color=color, alpha=0.15, linewidth=1)

        count += 1
        if count > 150: break

    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(-0.1, 0.6)
    ax.axis('off')

    # Custom Legend
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color=colors_sign['gain'], lw=2),
             Line2D([0], [0], color=colors_sign['loss'], lw=2)]
    ax.legend(lines, ['Gain', 'Loss'], loc='upper left', frameon=False)


if __name__ == '__main__':
    run_plotting()