# -*- coding: utf-8 -*-
"""
在原始 model.py 的基础上新增并实现图片中列出的三种双曲变体：
 1) 双曲 — 指数化分母：    IP = A / (1 + k * D) ** s
 2) 双曲 — 时间指数：        IP = A / (1 + k * (D ** s))
 3) 双曲 — 单位（乘积）指数：IP = A / (1 + (k * D) ** s)
同时保留原始的标准双曲和指数模型，完成拟合、R^2/AIC 比较与绘图。

使用方法：在包含数据 CSV 的目录下运行该脚本：
    python model_with_extra_hyperbolic_models.py

注意：A（立即收益）在本脚本中默认设为100（与原脚本一致）。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 第一步：数据预处理 --------------------------
def load_and_preprocess_data(csv_path):
    """加载CSV并预处理：筛选有效试次，按延迟分组"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    trial_df = df.dropna(subset=['trial.started']).copy()

    # 尝试安全地转换列类型（若不存在这些列会抛出错误）
    # 如果运行时出现列名不匹配，请根据你的 CSV 调整列名
    trial_df['standard_t_days'] = trial_df['standard_t_days'].astype(int)
    trial_df['comp_m_yuan'] = trial_df['comp_m_yuan'].astype(float)
    if 'rt' in trial_df.columns:
        trial_df = trial_df[(trial_df['rt'] > 0.2) & (trial_df['rt'] < 10)]

    print(f"有效试次数量：{len(trial_df)}")
    print(f"延迟区块分布：\n{trial_df['standard_t_days'].value_counts().sort_index()}")
    return trial_df

# -------------------------- 第二步：计算等价点（IP） --------------------------
def calculate_indifference_points(trial_df, n_last_trials=5):
    ip_data = []
    for delay in sorted(trial_df['standard_t_days'].unique()):
        delay_df = trial_df[trial_df['standard_t_days'] == delay].copy()
        delay_df = delay_df.sort_values('trial.started').reset_index(drop=True)
        if len(delay_df) >= n_last_trials:
            last_trials = delay_df.tail(n_last_trials)
            ip = last_trials['comp_m_yuan'].mean()
            ip_std = last_trials['comp_m_yuan'].std()
        else:
            ip = delay_df['comp_m_yuan'].mean()
            ip_std = delay_df['comp_m_yuan'].std()
        ip_data.append({
            'delay_days': delay,
            'indifference_point': ip,
            'ip_std': ip_std,
            'n_trials': len(delay_df)
        })
    ip_df = pd.DataFrame(ip_data)
    print("\n各延迟区块的等价点：")
    print(ip_df.round(2))
    return ip_df

# -------------------------- 第三步：定义贴现模型 --------------------------
A_default = 100.0  # 立即可得金额（与原脚本保持一致）

def hyperbolic_model(D, k):
    """标准双曲：A / (1 + k * D)"""
    return A_default / (1 + k * D)


def exponential_model(D, r):
    """指数贴现：A * exp(-r * D)"""
    return A_default * np.exp(-r * D)

# 新增三种变体（均为两参数 k, s 且 k>0, s>0）
def hyperbolic_exp_denominator(D, k, s):
    """分母整体指数化： A / (1 + k*D) ** s"""
    return A_default / np.power((1 + k * D), s)


def hyperbolic_time_exponent(D, k, s):
    """时间指数： A / (1 + k * (D ** s))"""
    return A_default / (1 + k * np.power(D, s))


def hyperbolic_unit_exponent(D, k, s):
    """乘积后再指数： A / (1 + (k * D) ** s)"""
    return A_default / (1 + np.power(k * D, s))

# -------------------------- 第四步：拟合模型并评估 --------------------------
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def fit_discount_models(ip_df):
    """拟合多种模型，返回参数、拟合曲线和评估指标"""
    X = ip_df['delay_days'].values
    y = ip_df['indifference_point'].values
    results = {}

    # 1. 标准双曲 (1 参数)
    try:
        popt_h, pcov_h = curve_fit(hyperbolic_model, X, y, bounds=(0, np.inf), maxfev=2000)
        k_h = popt_h[0]
        y_pred_h = hyperbolic_model(X, k_h)
        r2_h = r_squared(y, y_pred_h)
        perr_h = np.sqrt(np.diag(pcov_h)) if pcov_h is not None else [np.nan]
        results['hyper'] = {'params': {'k': k_h}, 'stderr': {'k': perr_h[0]}, 'y_pred': y_pred_h, 'r2': r2_h, 'n_params': 1}
    except Exception as e:
        print('标准双曲拟合失败：', e)

    # 2. 指数模型 (1 参数)
    try:
        popt_e, pcov_e = curve_fit(exponential_model, X, y, bounds=(0, np.inf), maxfev=2000)
        r_e = popt_e[0]
        y_pred_e = exponential_model(X, r_e)
        r2_e = r_squared(y, y_pred_e)
        perr_e = np.sqrt(np.diag(pcov_e)) if pcov_e is not None else [np.nan]
        results['exp'] = {'params': {'r': r_e}, 'stderr': {'r': perr_e[0]}, 'y_pred': y_pred_e, 'r2': r2_e, 'n_params': 1}
    except Exception as e:
        print('指数拟合失败：', e)

    # 为两参数模型准备 bounds 和初始猜测
    lower = [0.0, 0.01]     # k>=0, s>=0.01（避免 s==0 导致退化）
    upper = [np.inf, np.inf]
    p0 = [0.01, 1.0]

    # 3. 分母指数化模型 (k, s)
    try:
        popt_a, pcov_a = curve_fit(hyperbolic_exp_denominator, X, y, p0=p0, bounds=(lower, upper), maxfev=5000)
        k_a, s_a = popt_a
        y_pred_a = hyperbolic_exp_denominator(X, k_a, s_a)
        r2_a = r_squared(y, y_pred_a)
        perr_a = np.sqrt(np.diag(pcov_a))
        results['exp_denominator'] = {'params': {'k': k_a, 's': s_a}, 'stderr': {'k': perr_a[0], 's': perr_a[1]}, 'y_pred': y_pred_a, 'r2': r2_a, 'n_params': 2}
    except Exception as e:
        print('分母指数化模型拟合失败：', e)

    # 4. 时间指数模型 (k, s)
    try:
        popt_b, pcov_b = curve_fit(hyperbolic_time_exponent, X, y, p0=p0, bounds=(lower, upper), maxfev=5000)
        k_b, s_b = popt_b
        y_pred_b = hyperbolic_time_exponent(X, k_b, s_b)
        r2_b = r_squared(y, y_pred_b)
        perr_b = np.sqrt(np.diag(pcov_b))
        results['time_exponent'] = {'params': {'k': k_b, 's': s_b}, 'stderr': {'k': perr_b[0], 's': perr_b[1]}, 'y_pred': y_pred_b, 'r2': r2_b, 'n_params': 2}
    except Exception as e:
        print('时间指数模型拟合失败：', e)

    # 5. 单位（乘积）指数模型 (k, s)
    try:
        popt_c, pcov_c = curve_fit(hyperbolic_unit_exponent, X, y, p0=p0, bounds=(lower, upper), maxfev=5000)
        k_c, s_c = popt_c
        y_pred_c = hyperbolic_unit_exponent(X, k_c, s_c)
        r2_c = r_squared(y, y_pred_c)
        perr_c = np.sqrt(np.diag(pcov_c))
        results['unit_exponent'] = {'params': {'k': k_c, 's': s_c}, 'stderr': {'k': perr_c[0], 's': perr_c[1]}, 'y_pred': y_pred_c, 'r2': r2_c, 'n_params': 2}
    except Exception as e:
        print('单位（乘积）指数模型拟合失败：', e)

    # 计算 AIC 并输出摘要表格
    n = len(y)
    def compute_aic(y_true, y_pred, num_params):
        rss = np.sum((y_true - y_pred) ** 2)
        if rss <= 0:
            rss = 1e-12
        aic = n * np.log(rss / n) + 2 * num_params
        return aic

    print('\n模型拟合摘要：')
    rows = []
    for name, res in results.items():
        aic = compute_aic(y, res['y_pred'], res['n_params'])
        param_str = ', '.join([f"{k}={v:.6f}" for k, v in res['params'].items()])
        stderr_str = ', '.join([f"{k}±{v:.4f}" for k, v in res['stderr'].items()]) if 'stderr' in res else ''
        print(f"{name}: params: {param_str}; stderr: {stderr_str}; R2={res['r2']:.4f}; AIC={aic:.2f}")
        rows.append((name, res['r2'], aic))

    # 根据 AIC 排序
    rows_sorted = sorted(rows, key=lambda x: x[2])
    if rows_sorted:
        print('\n按 AIC 排序（越小越优）:')
        for r in rows_sorted:
            print(f"  {r[0]}: AIC={r[2]:.2f}, R2={r[1]:.4f}")

    return {'results': results, 'X': X, 'y': y}

# -------------------------- 第五步：可视化结果 --------------------------
def plot_results(ip_df, fit_dict):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimSun', 'FangSong', 'KaiTi']

    X = fit_dict['X']
    y = fit_dict['y']
    results = fit_dict['results']

    plt.figure(figsize=(10, 6))

    # 散点与误差棒
    plt.errorbar(ip_df['delay_days'], ip_df['indifference_point'], yerr=ip_df['ip_std'], fmt='o', markersize=8, capsize=4, label='等价点 (IP)')

    # 平滑绘制不同模型
    X_plot = np.linspace(0, max(ip_df['delay_days']) * 1.1, 300)

    for name, res in results.items():
        # 根据模型名调用对应函数以绘制平滑曲线
        if name == 'hyper':
            y_plot = hyperbolic_model(X_plot, res['params']['k'])
        elif name == 'exp':
            y_plot = exponential_model(X_plot, res['params']['r'])
        elif name == 'exp_denominator':
            y_plot = hyperbolic_exp_denominator(X_plot, res['params']['k'], res['params']['s'])
        elif name == 'time_exponent':
            y_plot = hyperbolic_time_exponent(X_plot, res['params']['k'], res['params']['s'])
        elif name == 'unit_exponent':
            y_plot = hyperbolic_unit_exponent(X_plot, res['params']['k'], res['params']['s'])
        else:
            continue
        label = f"{name} ({', '.join([f'{k}={v:.3g}' for k,v in res['params'].items()])}; R2={res['r2']:.3f})"
        plt.plot(X_plot, y_plot, linewidth=2, label=label)

    plt.xlabel('延迟天数 (Days)', fontsize=12)
    plt.ylabel('等价点 (Indifference Point, 元)', fontsize=12)
    plt.title('时间贴现模型比较拟合', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.xlim(-5, max(X_plot) + 5)
    plt.ylim(0, A_default + 5)
    plt.tight_layout()
    plt.show()

# -------------------------- 主流程 --------------------------
if __name__ == '__main__':
    csv_path = 'data\pilot_loop_2025-11-27_20h55.33.060.csv'  # 请根据实际文件路径修改
    trial_df = load_and_preprocess_data(csv_path)
    ip_df = calculate_indifference_points(trial_df, n_last_trials=5)
    fit_dict = fit_discount_models(ip_df)
    plot_results(ip_df, fit_dict)
