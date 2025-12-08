import os, json, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize
import glob
from scipy import stats

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

DATA_PATH = "data"  
FIG_DIR = "figs"
N_BINS = 8
MULTI_STARTS = 50  # 增加起点数
MIN_TRIALS_TO_FIT = 6
EPS_V = 1e-2
np.random.seed(42)
os.makedirs(FIG_DIR, exist_ok=True)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def save_fig(plt_obj, name):
    path = os.path.join(FIG_DIR, name)
    plt_obj.tight_layout()
    plt_obj.savefig(path, dpi=150)
    plt_obj.clf()

# 列检测
def detect_columns_strict(df):
    cols = list(df.columns)
    amt_cols = [c for c in cols if any(k in c.lower() for k in ("comp_m","amount"))]
    choice_cols = [c for c in cols if ("chose" in c.lower() or "choice" in c.lower())]
    rt_cols = [c for c in cols if c.lower()=="rt" or "reaction" in c.lower() or "response_time" in c.lower()]
    pressure_cols = [c for c in cols if "press" in c.lower() or "pressure" in c.lower()]
    time_cols = [c for c in cols if c.lower() in ("thisrow.t","t","trial_index")]
    delay_cols = [c for c in cols if (("delay" in c.lower() or c.lower().startswith("d_")) and "chose" not in c.lower())]
    return amt_cols, choice_cols, rt_cols, pressure_cols, time_cols, delay_cols

# 数据处理
def preprocess(df, file_name=None):
    amt_cols, choice_cols, rt_cols, pressure_cols, time_cols, delay_cols = detect_columns_strict(df)

    if len(amt_cols)==0:
        raise RuntimeError("No amount-like column found (comp_m expected).")
    amt_col = amt_cols[0]
    
    if len(choice_cols)==0:
        raise RuntimeError("No choice-like column found.")
    
    chosen_choice = None
    for c in choice_cols:
        if "chose" in c.lower():
            chosen_choice = c; break
    if chosen_choice is None:
        chosen_choice = choice_cols[0]
    
    rt_col = rt_cols[0] if len(rt_cols)>0 else None
    pressure_col = pressure_cols[0] if len(pressure_cols)>0 else None
    time_col = time_cols[0] if len(time_cols)>0 else None
    delay_col = delay_cols[0] if len(delay_cols)>0 else None

    work = df.copy()
    
    #文件标识
    if file_name:
        work['source_file'] = file_name
    
    #金额
    work['comp_m'] = pd.to_numeric(work[amt_col], errors='coerce')

    s = work[chosen_choice].astype(str).str.lower()
    mask_timeout = s == 'timeout'
    if mask_timeout.any():
        work = work[~mask_timeout].copy()
        s = work[chosen_choice].astype(str).str.lower()

    mapping = {'true':1.0, 'false':0.0, '1':1.0, '0':0.0}
    def map_choice(val):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        if v in mapping: return mapping[v]
        try:
            return float(v)
        except:
            return np.nan
    work['choice'] = work[chosen_choice].apply(map_choice)

    if rt_col is not None:
        work['rt'] = pd.to_numeric(work[rt_col], errors='coerce')
        work.loc[work['rt']<=0,'rt'] = np.nan
    else:
        work['rt'] = np.nan

    if pressure_col is not None:
        work['block_pressure'] = work[pressure_col].astype(str)
    else:
        work['block_pressure'] = 'no_pressure'

    if time_col is not None:
        work['t'] = pd.to_numeric(work[time_col], errors='coerce')
        if work['t'].isna().all():
            if 'participant' in work.columns:
                work['t'] = work.groupby(['participant','block_pressure']).cumcount()
            else:
                work['t'] = np.arange(len(work))
    else:
        if 'participant' in work.columns:
            work['t'] = work.groupby(['participant','block_pressure']).cumcount()
        else:
            work['t'] = np.arange(len(work))

    if delay_col is not None:
        work['delay_attr'] = pd.to_numeric(work[delay_col], errors='coerce')
    else:
        work['delay_attr'] = np.nan

    # 过滤缺失
    before = len(work)
    work = work[~work['comp_m'].isna() & ~work['choice'].isna()].copy()

    return work

# ========== 修改权重函数形式 ==========
# 原函数: w_amt_t(t, alpha, lam, c): return alpha * np.exp(-lam * t) + c
# 修改为更灵活的形式，增加线性项和幂律项

# 方案1: 指数衰减 + 线性趋势
def w_amt_t_v1(t, alpha, lam, beta, c):
    """
    指数衰减 + 线性趋势
    alpha: 初始权重
    lam: 衰减率
    beta: 线性趋势系数
    c: 基础偏移
    """
    return alpha * np.exp(-lam * t) + beta * t + c

# 方案2: 双指数衰减（更灵活）
def w_amt_t_v2(t, alpha1, lam1, alpha2, lam2, c):
    """
    双指数衰减，可以模拟更复杂的衰减模式
    alpha1, lam1: 第一个指数项
    alpha2, lam2: 第二个指数项
    c: 基础偏移
    """
    return alpha1 * np.exp(-lam1 * t) + alpha2 * np.exp(-lam2 * t) + c

# 方案3: 指数衰减 + 幂律项
def w_amt_t_v3(t, alpha, lam, beta, gamma, c):
    """
    指数衰减 + 幂律项
    alpha: 指数项系数
    lam: 衰减率
    beta: 幂律项系数
    gamma: 幂律指数
    c: 基础偏移
    """
    return alpha * np.exp(-lam * t) + beta * (t + 1)**(-gamma) + c

# 使用方案1作为默认（最简单有效的扩展）
w_amt_t = w_amt_t_v1

# 对应的时间权重函数也需要扩展（如果使用full模型）
def w_time_t(t, alpha, t0, tau, c):
    return alpha / (1.0 + np.exp(-(t - t0)/(tau+1e-12))) + c

# ========== 修改预测与损失函数 ==========
# simplified模型现在有6个参数: [alpha, lam, beta, c, bias, k_choice]

def predict_v_simplified(params, t, A):
    alpha, lam, beta, c, bias, k_choice = params
    w1 = w_amt_t(t, alpha, lam, beta, c)
    v = w1 * A + bias
    return v, w1

def loss_simplified(params, t, A, Y, reg=1e-3, trend_weight=0.05):
    v, w1 = predict_v_simplified(params, t, A)
    if np.isnan(v).any() or not np.isfinite(v).all():
        return 1e8
    
    logits = params[5] * v  # 注意索引现在是5
    logits = np.clip(logits, -100, 100)
    p = sigmoid(logits)
    eps = 1e-9
    
    # 负对数似然
    nll = -np.sum(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))
    
    # L2正则化
    l2 = reg * np.sum(np.square(params[:5]))  # 前5个参数
    
    # 趋势惩罚项：鼓励权重随时间变化（但不过度）
    w_grad = np.gradient(w1)
    trend_penalty = trend_weight * np.mean(np.abs(w_grad))
    
    # 防止beta过大导致爆炸性变化
    beta = params[2]
    if np.abs(beta) > 0.1:
        trend_penalty += 0.1 * np.abs(beta)
    
    return nll + l2 + trend_penalty

# full模型的预测函数需要相应调整
def predict_v_full(params, t, A, D):
    # 注意：full模型现在有10个参数
    # [alpha_a, lam_a, beta_a, c_a, alpha_t, t0, tau, c_t, bias, k_choice]
    alpha_a, lam_a, beta_a, c_a, alpha_t, t0, tau, c_t, bias, k_choice = params
    w1 = w_amt_t(t, alpha_a, lam_a, beta_a, c_a)
    w2 = w_time_t(t, alpha_t, t0, tau, c_t)
    v = w1 * A + w2 * D + bias
    return v, w1, w2

def loss_full(params, t, A, D, Y, reg=1e-3):
    v, w1, w2 = predict_v_full(params, t, A, D)
    if np.isnan(v).any() or not np.isfinite(v).all():
        return 1e8
    
    logits = params[9] * v  # 注意索引现在是9
    logits = np.clip(logits, -100, 100)
    p = sigmoid(logits)
    eps = 1e-9
    
    nll = -np.sum(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))
    l2 = reg * np.sum(np.square(params[:9]))  # 前9个参数
    return nll + l2

# ========== 修改拟合函数 ==========
def fit_simplified_multistart(t, A, Y, nstarts=MULTI_STARTS):
    best = None
    best_loss = 1e20
    
    # 分析数据的时间趋势
    try:
        # 计算时间与选择的相关性
        if len(t) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, Y)
            print(f"  数据趋势: 斜率={slope:.4f}, R={r_value:.3f}, p={p_value:.4f}")
            
            # 基于趋势调整初始化策略
            if slope < -0.05:  # 明显的下降趋势
                print(f"  检测到下降趋势，增加衰减参数的权重")
        else:
            slope = 0.0
    except:
        slope = 0.0
    
    for i in range(nstarts):
        # 根据数据特征调整初始化策略
        if slope < -0.05:  # 下降趋势明显
            if i % 3 == 0:
                # 快速衰减模式
                init = np.array([
                    np.random.uniform(0.5, 2.5),  # alpha
                    np.random.uniform(0.05, 0.5), # lam (较大的衰减率)
                    np.random.uniform(-0.01, -0.001), # beta (负的线性项)
                    np.random.uniform(-0.5, 0.5), # c
                    np.random.uniform(-1, 1),     # bias
                    np.random.uniform(0.5, 3.0)   # k_choice
                ])
            elif i % 3 == 1:
                # 中等衰减模式
                init = np.array([
                    np.random.uniform(0.3, 1.5),
                    np.random.uniform(0.01, 0.2),
                    np.random.uniform(-0.005, 0),
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(1.0, 5.0)
                ])
            else:
                # 随机模式
                init = np.array([
                    np.random.uniform(0.2, 3.0),
                    np.random.uniform(1e-6, 0.8),
                    np.random.uniform(-0.02, 0.02),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-2, 2),
                    np.random.uniform(0.1, 10.0)
                ])
        else:
            # 没有明显趋势
            init = np.array([
                np.random.uniform(0.2, 3.0),    # alpha
                np.random.uniform(1e-6, 0.5),   # lam
                np.random.uniform(-0.01, 0.01), # beta (接近0)
                np.random.uniform(-0.5, 0.5),   # c
                np.random.uniform(-1, 1),       # bias
                np.random.uniform(0.5, 5.0)     # k_choice
            ])
        
        # 参数边界
        bounds = [
            (1e-6, 10),      # alpha
            (1e-8, 2.0),     # lam (允许更大的衰减率)
            (-0.1, 0.1),     # beta (线性项系数，不要太大)
            (-2, 2),         # c
            (-3, 3),         # bias
            (1e-3, 20)       # k_choice
        ]
        
        try:
            res = minimize(
                lambda p: loss_simplified(p, t, A, Y),
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-08}
            )
            
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best = res.x
        except Exception as e:
            continue
    
    return best, best_loss

def fit_full_multistart(t, A, D, Y, nstarts=MULTI_STARTS):
    best = None
    best_loss = 1e20
    
    for i in range(nstarts):
        init = np.array([
            np.random.uniform(0.2, 3.0),    # alpha_a
            np.random.uniform(1e-6, 0.5),   # lam_a
            np.random.uniform(-0.01, 0.01), # beta_a
            np.random.uniform(-0.5, 0.5),   # c_a
            np.random.uniform(-1, 1),       # alpha_t
            np.median(t),                   # t0
            np.random.uniform(1.0, 20.0),   # tau
            np.random.uniform(-0.5, 0.5),   # c_t
            np.random.uniform(-1, 1),       # bias
            np.random.uniform(0.5, 5.0)     # k_choice
        ])
        
        bounds = [
            (1e-6, 10),      # alpha_a
            (1e-8, 2.0),     # lam_a
            (-0.1, 0.1),     # beta_a
            (-2, 2),         # c_a
            (-5, 5),         # alpha_t
            (np.min(t)-50, np.max(t)+50),  # t0
            (1e-3, 50),      # tau
            (-5, 5),         # c_t
            (-5, 5),         # bias
            (1e-3, 20)       # k_choice
        ]
        
        try:
            res = minimize(
                lambda p: loss_full(p, t, A, D, Y),
                init,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                best = res.x
        except:
            continue
    
    return best, best_loss

def fit_rt_given_v(v, RT):
    mask = ~np.isnan(RT) & np.isfinite(RT)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = 1.0/(np.abs(v)+EPS_V)
    X = np.vstack([np.ones(mask.sum()), x[mask]]).T
    y = RT[mask]
    coef, *_ = np.linalg.lstsq(X,y,rcond=None)
    return float(coef[0]), float(coef[1])

# ========== 添加诊断函数 ==========
def diagnose_fit(t, A, Y, p_hat, params, scope_name):
    """诊断模型拟合质量"""
    print(f"\n=== 模型诊断: {scope_name} ===")
    
    # 1. 计算预测准确率
    pred_choice = (p_hat > 0.5).astype(float)
    accuracy = np.mean(pred_choice == Y)
    print(f"预测准确率: {accuracy:.3f}")
    
    # 2. 检查时间趋势
    n_bins = min(10, len(t))
    if n_bins >= 3:
        t_bins = np.linspace(np.min(t), np.max(t), n_bins + 1)
        idx = np.digitize(t, t_bins) - 1
        idx[idx < 0] = 0
        idx[idx >= n_bins] = n_bins - 1
        
        actual_means = []
        pred_means = []
        for i in range(n_bins):
            mask = (idx == i)
            if np.sum(mask) > 0:
                actual_means.append(np.mean(Y[mask]))
                pred_means.append(np.mean(p_hat[mask]))
        
        if len(actual_means) > 2:
            # 计算趋势相关性
            actual_trend = np.array(actual_means)
            pred_trend = np.array(pred_means)
            
            # 计算斜率
            x_range = np.arange(len(actual_trend))
            slope_actual, _ = np.polyfit(x_range, actual_trend, 1)
            slope_pred, _ = np.polyfit(x_range, pred_trend, 1)
            
            print(f"实际选择率趋势: {slope_actual:.4f} (每bin变化)")
            print(f"预测选择率趋势: {slope_pred:.4f} (每bin变化)")
            print(f"趋势方向匹配: {'是' if slope_actual * slope_pred > 0 else '否'}")
    
    # 3. 检查参数合理性
    alpha, lam, beta, c, bias, k_choice = params
    print(f"权重参数: α={alpha:.4f}, λ={lam:.4f}, β={beta:.6f}, c={c:.4f}")
    print(f"决策参数: bias={bias:.4f}, k={k_choice:.4f}")
    
    # 4. 计算权重变化范围
    w1 = w_amt_t(t, alpha, lam, beta, c)
    print(f"权重变化范围: {np.min(w1):.4f} 到 {np.max(w1):.4f}")
    print(f"权重相对变化: {(np.max(w1)-np.min(w1))/np.mean(w1):.2%}")
    
    # 5. 可视化诊断
    plt.figure(figsize=(12, 4))
    
    # 子图1: 实际vs预测散点
    plt.subplot(1, 3, 1)
    plt.scatter(t, Y, alpha=0.3, s=15, label='实际', color='blue')
    sorted_idx = np.argsort(t)
    plt.plot(t[sorted_idx], p_hat[sorted_idx], 'r-', linewidth=2, label='预测')
    plt.xlabel('试次 (t)')
    plt.ylabel('选择概率')
    plt.title('实际vs预测散点')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 权重函数
    plt.subplot(1, 3, 2)
    t_grid = np.linspace(np.min(t), np.max(t), 200)
    w_grid = w_amt_t(t_grid, alpha, lam, beta, c)
    plt.plot(t_grid, w_grid, 'g-', linewidth=2)
    plt.xlabel('试次 (t)')
    plt.ylabel('权重 w(t)')
    plt.title(f'权重函数: α={alpha:.3f}, λ={lam:.3f}, β={beta:.5f}')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 残差图
    plt.subplot(1, 3, 3)
    residuals = Y - p_hat
    plt.scatter(t, residuals, alpha=0.5, s=15)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('试次 (t)')
    plt.ylabel('残差 (实际-预测)')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(plt, f"{scope_name.replace('/', '_')}_diagnosis.png")

def plot_obs_pred_binned(t, Y, p_hat, scope_name, fname):
    bins = np.linspace(np.min(t), np.max(t), N_BINS+1)
    idx = np.digitize(t, bins)-1
    idx[idx<0]=0; idx[idx>=N_BINS]=N_BINS-1
    centers = 0.5*(bins[:-1]+bins[1:])
    obs = [np.nan if not np.any(idx==i) else np.nanmean(Y[idx==i]) for i in range(N_BINS)]
    pred= [np.nan if not np.any(idx==i) else np.nanmean(p_hat[idx==i]) for i in range(N_BINS)]
    plt.figure(figsize=(8,5))
    plt.plot(centers, obs, 'bo-', linewidth=2, markersize=8, label='实际')
    plt.plot(centers, pred, 'ro-', linewidth=2, markersize=8, label='预测')
    plt.ylim(-0.05,1.05); plt.xlabel('试次 (t)'); plt.ylabel('选择延迟概率')
    plt.title(f"{scope_name}: 实际vs预测 (分箱)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(plt, fname)

def plot_weight_curve(t, w_func_vals, label, scope_name, fname):
    plt.figure(figsize=(8,4))
    plt.plot(t, w_func_vals, 'g-', linewidth=2, label=label)
    plt.title(f"{scope_name}: {label}")
    plt.xlabel('试次 (t)'); plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(plt, fname)

# 整合全部csv
def load_all_data(data_folder):
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {data_folder}")
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    
    all_data = []
    
    for file_path in csv_files:
        try:
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path)
            processed_df = preprocess(df, file_name)
            all_data.append(processed_df)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    if not all_data:
        raise RuntimeError("无法加载任何数据，请检查 CSV 文件格式")
    
    #合并
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n合并后总数据: {len(combined_data)} 行")
    print(f"来源文件数: {combined_data['source_file'].nunique()}")
    print(f"压力条件类型: {combined_data['block_pressure'].unique()}")
    
    return combined_data

def main():
    data = load_all_data(DATA_PATH)

    scopes = [("all", data)]
    
    # 按压力条件分组
    for blk in data['block_pressure'].unique():
        blk_data = data[data['block_pressure'] == blk].copy()
        scopes.append((f"pressure_{blk}", blk_data))

    out = {}
    for scope_name, df in scopes:
        print("\n" + "=" * 60)
        print(f"=== 分析范围: {scope_name}, 样本量={len(df)} ===")
        
        if len(df) < MIN_TRIALS_TO_FIT:
            print("试次太少，跳过")
            out[scope_name] = {"fitted": False, "reason":"too_few"}
            continue

        t = pd.to_numeric(df['t'], errors='coerce').values.astype(float)
        A = pd.to_numeric(df['comp_m'], errors='coerce').values.astype(float)
        D = pd.to_numeric(df['delay_attr'], errors='coerce').values.astype(float)
        Y = pd.to_numeric(df['choice'], errors='coerce').values.astype(float)
        RT = pd.to_numeric(df['rt'], errors='coerce').values.astype(float)

        # 更谨慎的标准化
        # 对金额：只做简单的归一化
        A_min, A_max = np.nanmin(A), np.nanmax(A)
        if A_max > A_min:
            A_v = (A - A_min) / (A_max - A_min)
        else:
            A_v = A.copy()
        
        # 对时间：保持原始尺度，只做中心化
        if np.isnan(t).all():
            t0 = np.arange(len(t))
        else:
            t0 = t - np.nanmin(t)
            # 除非时间尺度非常大，否则不缩放
            if np.nanmax(t0) > 5000:
                t0 = t0 / 100.0
                print(f"  时间尺度较大，缩放为1/100")
        
        # 检查延迟信息
        if np.isnan(D).all():
            D_present = False
        else:
            D_filled = np.nan_to_num(D, nan=0.0)
            D_mean = np.nanmean(D_filled); D_std = np.nanstd(D_filled)+1e-8
            D_v = (D_filled - D_mean)/D_std
            D_present = True

        # 应用mask
        mask = (~np.isnan(A_v)) & (~np.isnan(Y)) & (~np.isnan(t0))
        t_v = t0[mask]; A_v = A_v[mask]; Y_v = Y[mask]; RT_v = RT[mask]
        D_v = D_v[mask] if D_present else None
        print(f"有效试次: {len(t_v)} (过滤后)")
        
        # 分析实际数据的趋势
        if len(t_v) > 10:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(t_v, Y_v)
                print(f"  实际数据趋势: 斜率={slope:.6f}, R={r_value:.3f}, p={p_value:.4f}")
            except:
                pass

        # 金额分布图
        plt.figure(figsize=(6,3))
        plt.hist(A[~np.isnan(A)], bins=12, edgecolor='black', alpha=0.7)
        plt.title(f"{scope_name} 金额分布 (n={np.sum(~np.isnan(A))})")
        plt.xlabel('金额')
        plt.ylabel('频次')
        save_fig(plt, f"{scope_name.replace('/', '_')}_compm_hist.png")

        if len(t_v) < MIN_TRIALS_TO_FIT:
            print("有效试次不足，跳过拟合")
            out[scope_name] = {"fitted": False, "reason":"not_enough_after_mask"}
            continue

        if not D_present:
            print("拟合简化模型 (仅金额)")
            params_best, loss_best = fit_simplified_multistart(t_v, A_v, Y_v, nstarts=MULTI_STARTS)
            
            if params_best is None:
                print("拟合失败")
                out[scope_name] = {"fitted": False, "reason":"fitting_failed"}
                continue
                
            print(f"最佳参数: {params_best}")
            print(f"损失值: {loss_best:.4f}")
            
            v_hat, w1 = predict_v_simplified(params_best, t_v, A_v)
            p_hat = sigmoid(params_best[5] * v_hat)
            
            # 诊断
            diagnose_fit(t_v, A_v, Y_v, p_hat, params_best, scope_name)
            
            # 绘制分箱图
            plot_obs_pred_binned(t_v, Y_v, p_hat, scope_name, 
                               f"{scope_name.replace('/', '_')}_obs_vs_pred.png")
            
            # 绘制权重曲线
            tgrid = np.linspace(np.min(t_v), np.max(t_v), 300)
            wgrid = w_amt_t(tgrid, params_best[0], params_best[1], 
                           params_best[2], params_best[3])
            plot_weight_curve(tgrid, wgrid, "w_amt(t)", scope_name, 
                            f"{scope_name.replace('/', '_')}_w_amt.png")
            
            # RT拟合
            ndt, krt = fit_rt_given_v(v_hat, RT_v)
            if not np.isnan(ndt):
                rt_pred = ndt + krt/(np.abs(v_hat)+EPS_V)
                plt.figure(figsize=(5,5))
                plt.scatter(RT_v, rt_pred, alpha=0.7)
                mn, mx = np.nanmin(RT_v), np.nanmax(RT_v)
                plt.plot([mn,mx],[mn,mx],'k--')
                plt.title(f"{scope_name} RT 实际vs预测")
                plt.xlabel('实际RT')
                plt.ylabel('预测RT')
                save_fig(plt, f"{scope_name.replace('/', '_')}_rt_obs_pred.png")
            
            out[scope_name] = {
                "fitted": True, 
                "model": "simplified_v1",  # 标注使用的新模型版本
                "params": {
                    "alpha_amt": float(params_best[0]), 
                    "lam_amt": float(params_best[1]),
                    "beta_amt": float(params_best[2]),
                    "c_amt": float(params_best[3]), 
                    "bias": float(params_best[4]), 
                    "k_choice": float(params_best[5]),
                    "loss": float(loss_best), 
                    "ndt": None if np.isnan(ndt) else float(ndt), 
                    "k_rt": None if np.isnan(krt) else float(krt)
                }
            }
        else:
            print("拟合完整模型 (金额 + 延迟)")
            params_best, loss_best = fit_full_multistart(t_v, A_v, D_v, Y_v, nstarts=MULTI_STARTS)
            
            if params_best is None:
                print("拟合失败")
                out[scope_name] = {"fitted": False, "reason":"fitting_failed"}
                continue
                
            print(f"最佳参数: {params_best}")
            print(f"损失值: {loss_best:.4f}")
            
            v_hat, w1, w2 = predict_v_full(params_best, t_v, A_v, D_v)
            p_hat = sigmoid(params_best[9] * v_hat)
            
            # 绘制分箱图
            plot_obs_pred_binned(t_v, Y_v, p_hat, scope_name, 
                               f"{scope_name.replace('/', '_')}_obs_vs_pred.png")
            
            # 绘制权重曲线
            tgrid = np.linspace(np.min(t_v), np.max(t_v), 300)
            w1grid = w_amt_t(tgrid, params_best[0], params_best[1], 
                           params_best[2], params_best[3])
            w2grid = w_time_t(tgrid, params_best[4], params_best[5], 
                            params_best[6], params_best[7])
            plot_weight_curve(tgrid, w1grid, "w_amt(t)", scope_name, 
                            f"{scope_name.replace('/', '_')}_w_amt.png")
            plot_weight_curve(tgrid, w2grid, "w_time(t)", scope_name, 
                            f"{scope_name.replace('/', '_')}_w_time.png")
            
            # RT拟合
            ndt, krt = fit_rt_given_v(v_hat, RT_v)
            if not np.isnan(ndt):
                rt_pred = ndt + krt/(np.abs(v_hat)+EPS_V)
                plt.figure(figsize=(5,5))
                plt.scatter(RT_v, rt_pred, alpha=0.7)
                mn, mx = np.nanmin(RT_v), np.nanmax(RT_v)
                plt.plot([mn,mx],[mn,mx],'k--')
                plt.title(f"{scope_name} RT 实际vs预测")
                plt.xlabel('实际RT')
                plt.ylabel('预测RT')
                save_fig(plt, f"{scope_name.replace('/', '_')}_rt_obs_pred.png")
            
            out[scope_name] = {
                "fitted": True, 
                "model": "full_v1",  # 标注使用的新模型版本
                "params": params_best.tolist(), 
                "loss": float(loss_best)
            }
    
    print("完成")

if __name__=='__main__':
    main()