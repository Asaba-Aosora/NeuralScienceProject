# 由于psychopy里用的PEST实验算法，最后会自动收敛到等价点，所以取每个区块的最后3-5个试次取平均、作为等价点

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 第一步：数据预处理 --------------------------
def load_and_preprocess_data(csv_path):
    """加载CSV并预处理：筛选有效试次，按延迟分组"""
    # 加载数据
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 筛选试次数据（排除Intro和end的行，只保留有trial.started的试次）
    trial_df = df.dropna(subset=['trial.started']).copy()
    
    # 转换数据类型
    trial_df['standard_t_days'] = trial_df['standard_t_days'].astype(int)
    trial_df['comp_m_yuan'] = trial_df['comp_m_yuan'].astype(float)
    trial_df['reversal_count'] = trial_df['reversal_count'].astype(int)
    
    # 可选：过滤异常反应时（RT<0.2s或>10s，本数据无异常）
    trial_df = trial_df[(trial_df['rt'] > 0.2) & (trial_df['rt'] < 10)]
    
    print(f"有效试次数量：{len(trial_df)}")
    print(f"延迟区块分布：\n{trial_df['standard_t_days'].value_counts().sort_index()}")
    
    return trial_df

# -------------------------- 第二步：计算等价点（IP） --------------------------
def calculate_indifference_points(trial_df, n_last_trials=5):
    """计算每个延迟区块的等价点：取最后n次试次的comp_m_yuan平均值"""
    ip_data = []
    
    # 按延迟分组计算等价点
    for delay in sorted(trial_df['standard_t_days'].unique()):
        # 该延迟的所有试次
        delay_df = trial_df[trial_df['standard_t_days'] == delay].copy()
        
        # 按试次顺序排序（按trial.started时间）
        delay_df = delay_df.sort_values('trial.started').reset_index(drop=True)
        
        # 取最后n_last_trials个试次（步长最小，逼近等价点）
        if len(delay_df) >= n_last_trials:
            last_trials = delay_df.tail(n_last_trials)
            ip = last_trials['comp_m_yuan'].mean()
            ip_std = last_trials['comp_m_yuan'].std()  # 等价点标准差（可靠性指标）
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
def hyperbolic_model(D, k):
    """双曲贴现模型：IP = 100 / (1 + k*D)"""
    A = 100  # 实验中固定的延迟金额
    return A / (1 + k * D)

def exponential_model(D, r):
    """指数贴现模型：IP = 100 * exp(-r*D)"""
    A = 100
    return A * np.exp(-r * D)

# -------------------------- 第四步：拟合模型并评估 --------------------------
def fit_discount_models(ip_df):
    """拟合双曲和指数模型，返回参数和拟合优度"""
    # 提取拟合数据（排除D=0？可选，D=0时理论IP=100，用于验证）
    X = ip_df['delay_days'].values  # 延迟天数
    y = ip_df['indifference_point'].values  # 等价点
    
    # 拟合双曲模型（k>0）
    popt_hyper, pcov_hyper = curve_fit(
        f=hyperbolic_model,
        xdata=X,
        ydata=y,
        bounds=(0, np.inf),  # k必须为正
        maxfev=1000
    )
    k_estimate = popt_hyper[0]
    y_pred_hyper = hyperbolic_model(X, k_estimate)
    
    # 拟合指数模型（r>0）
    popt_exp, pcov_exp = curve_fit(
        f=exponential_model,
        xdata=X,
        ydata=y,
        bounds=(0, np.inf),  # r必须为正
        maxfev=1000
    )
    r_estimate = popt_exp[0]
    y_pred_exp = exponential_model(X, r_estimate)
    
    # 计算拟合优度R²（1 - 残差平方和/总平方和）
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
    
    r2_hyper = r_squared(y, y_pred_hyper)
    r2_exp = r_squared(y, y_pred_exp)
    
    # 输出结果
    print("\n模型拟合结果：")
    print(f"双曲贴现模型 - 贴现率k = {k_estimate:.6f}, R² = {r2_hyper:.4f}")
    print(f"指数贴现模型 - 贴现率r = {r_estimate:.6f}, R² = {r2_exp:.4f}")
    
    # 比较AIC（越小越好，惩罚参数数量）
    n = len(y)  # 样本数
    k_params_hyper = 1  # 双曲模型1个参数（k）
    k_params_exp = 1    # 指数模型1个参数（r）
    
    aic_hyper = n * np.log(np.sum((y - y_pred_hyper)**2)/n) + 2 * k_params_hyper
    aic_exp = n * np.log(np.sum((y - y_pred_exp)**2)/n) + 2 * k_params_exp
    
    print(f"AIC比较 - 双曲模型：{aic_hyper:.2f}, 指数模型：{aic_exp:.2f}")
    print(f"最优模型：{'双曲模型' if aic_hyper < aic_exp else '指数模型'}")
    
    return {
        'hyper': {'k': k_estimate, 'y_pred': y_pred_hyper, 'r2': r2_hyper},
        'exp': {'r': r_estimate, 'y_pred': y_pred_exp, 'r2': r2_exp},
        'X': X, 'y': y
    }

# -------------------------- 第五步：可视化结果 --------------------------
def plot_results(ip_df, fit_results):
    """绘制等价点散点+模型拟合曲线"""
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimSun', 'FangSong', 'KaiTi']

    plt.figure(figsize=(10, 6))
    
    # 绘制等价点散点（带误差棒）
    plt.errorbar(
        ip_df['delay_days'], ip_df['indifference_point'],
        yerr=ip_df['ip_std'], fmt='o', color='black', markersize=8,
        capsize=5, label='等价点（IP）± 标准差'
    )
    
    # 绘制拟合曲线（扩展X轴范围，使曲线更完整）
    X_plot = np.linspace(0, max(ip_df['delay_days']) * 1.1, 100)
    y_hyper_plot = hyperbolic_model(X_plot, fit_results['hyper']['k'])
    y_exp_plot = exponential_model(X_plot, fit_results['exp']['r'])
    
    plt.plot(X_plot, y_hyper_plot, 'r-', linewidth=2, 
             label=f'双曲模型 (k={fit_results["hyper"]["k"]:.6f}, R²={fit_results["hyper"]["r2"]:.4f})')
    plt.plot(X_plot, y_exp_plot, 'b--', linewidth=2,
             label=f'指数模型 (r={fit_results["exp"]["r"]:.6f}, R²={fit_results["exp"]["r2"]:.4f})')
    
    # 图表设置
    plt.xlabel('延迟天数 (Days)', fontsize=12)
    plt.ylabel('等价点 (Indifference Point, 元)', fontsize=12)
    plt.title('时间贴现模型拟合结果', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(-10, max(X_plot) + 10)
    plt.ylim(0, 110)  # 等价点不超过100
    plt.tight_layout()
    plt.show()

# -------------------------- 主函数：串联所有步骤 --------------------------
if __name__ == '__main__':
    # 1. 加载并预处理数据（替换为你的CSV路径）
    csv_path = 'data/030330_loop_2025-11-27_20h25.06.445.csv'
    trial_df = load_and_preprocess_data(csv_path)
    
    # 2. 计算等价点
    ip_df = calculate_indifference_points(trial_df, n_last_trials=5)
    
    # 3. 拟合模型
    fit_results = fit_discount_models(ip_df)
    
    # 4. 可视化
    plot_results(ip_df, fit_results)

