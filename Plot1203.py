"""
Project: Trajectory of Hesitation - Advanced Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
from scipy.stats import gaussian_kde  
warnings.filterwarnings('ignore')

#显示设置
sns.set_theme(style="ticks", context="talk", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # Standard English fonts

#颜色设置
colors_pressure = {'no_pressure': '#2E86C1', 'high_pressure': "#e15a4b"} 
colors_sign = {'gain': '#F1C40F', 'loss': '#8E44AD'}  

# 数据加载与预处理
def load_data():
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
            # 基础清洗
            clean_df = full_df[full_df['rt'] > 0.1].copy()
            #手动转换选择列为整数
            if clean_df['chose_delayed'].dtype == object:
                clean_df['chose_delayed_int'] = clean_df['chose_delayed'].astype(str).map(
                    {'True': 1, 'False': 0, '1': 1, '0': 0})
            else:
                clean_df['chose_delayed_int'] = clean_df['chose_delayed'].astype(int)
            return clean_df

    print("No data found or load failed. ")



# 主绘图函数
def run_plotting():
    df = load_data()

    calculate_main_effect(df)
    
    # 预计算
    # IP
    df_ip = df.groupby(['participant', 'block_sign', 'block_pressure']).tail(3)
    df_ip_agg = df_ip.groupby(['participant', 'block_sign', 'block_pressure'])['comp_m'].mean().reset_index()

    # Difficulty & Bins
    df_merged = df.merge(df_ip_agg, on=['participant', 'block_sign', 'block_pressure'], suffixes=('', '_final'))
    df_merged['difficulty'] = abs(df_merged['comp_m'] - df_merged['comp_m_final'])
    df_merged['diff_bin'] = pd.cut(df_merged['difficulty'], bins=[-1, 5, 20, 100], labels=['Hard', 'Medium', 'Easy'])

    # 计算Real Value Difference
    # 估计 Value Diff 
    df_merged['val_diff_real'] = df_merged['comp_m_final'] - df_merged['comp_m']

    print(">>> Generating Plots...")


    # Figure 1: Indifference Points (Behavioral Outcome)
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


    # Figure 2: Chronometric Function (DDM Mechanism)
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


    # Figure 3: Trajectory Max Deviation (Motor Conflict)
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

    # Figure 4: Psychometric Curve (Sensitivity Analysis)
    # 展示选择概率如何随价值差变化。高压组斜率更平，说明敏感度下降。
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

    # Figure 5: RT Distribution (Collapsed Bound Evidence)
    # 直观展示高压如何把RT分布“挤压”到左侧。
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

    # Figure 6: Motor-Decision Correlation (MD vs RT)
    # 验证“思维泄露”。越难的决策(RT长)，手抖得越厉害(MD大)。
    g = sns.jointplot(data=df_merged, x="rt", y="max_deviation", hue="block_pressure",
                      palette=colors_pressure, alpha=0.4, height=8)

    g.fig.suptitle("Fig 6. Correlation: Decision Time vs. Motor Conflict", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Reaction Time (s)", "Max Deviation (Curvature)")
    plt.show()


from statsmodels.stats.anova import AnovaRM

def calculate_main_effect(df):
    print("\n========== 压力主效应计算 (ANOVA) ==========")
    
    # 准备数据，先计算每个Block的最终IP（取平均）
    df_ip = df.groupby(['participant', 'block_sign', 'block_pressure'])['comp_m'].mean().reset_index()
    
    #重复测量方差分析 
    aov = AnovaRM(
        data=df_ip, 
        depvar='comp_m', 
        subject='participant', 
        within=['block_sign', 'block_pressure']
    )
    res = aov.fit()
    
    print(res)
    

if __name__ == '__main__':
    run_plotting()