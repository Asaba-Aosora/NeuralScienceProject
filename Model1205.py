"""
项目名称：犹豫的轨迹 (Trajectory of Hesitation)
文件：analysis_script_advanced.py
功能：深度数据分析。使用分层统计模型（GLMM/LME）估计鲁棒的 k 值，并分析轨迹指标。
依赖：pandas, numpy, scipy, statsmodels (用于 GLMM/LME)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import os

# --- 1. 配置与数据加载 ---

# !!! 请修改此路径指向您实际的 CSV 文件!!!
FILE_PATH = 'data/your_participant_id_Mouse_Track_Study_DATE_TIME.csv'

DAYS_DELAY = 7  # 实验中固定延迟天数 D
STANDARD_AMOUNT = 100  # 实验中固定标准金额 A


# --- 2. 轨迹处理辅助函数 ---

def parse_trajectory(traj_str):
    """将轨迹字符串 (x1,y1;x2,y2...) 解析为 (x, y) 列表"""
    if not isinstance(traj_str, str) or traj_str == 'timeout':
        return np.array(), np.array()

    points = [tuple(map(float, p.split(','))) for p in traj_str.split(';') if p]
    if not points:
        return np.array(), np.array()

    x = np.array([p for p in points])
    y = np.array([p[1] for p in points])
    return x, y


def normalize_and_calculate_auc(x_raw, y_raw, choice_side):
    """
    进行空间和时间规范化，并计算 AUC (Area Under the Curve)
    AUC 是比 MD 更全面的冲突指标。
    """

    if len(x_raw) < 5:  # 轨迹点太少，无法计算
        return 0.0, 0.0

    # 定义起点和终点
    start_p = np.array([0.0, -0.4])
    end_p = np.array([-0.4, 0.3]) if choice_side == 'left' else np.array([0.4, 0.3])

    # 空间归一化 (将所有轨迹映射到统一的目标空间，例如，左目标在 (-1, 1), 右目标在 (1, 1))
    # 1. 旋转和平移：使起点为 (0,0)，理想直线在 x 轴上

    # 理想直线向量
    ideal_vec = end_p - start_p
    theta = np.arctan2(ideal_vec[1], ideal_vec)

    # 旋转矩阵
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])

    # 移动到原点并旋转
    points_centered = np.array(list(zip(x_raw, y_raw))) - start_p
    points_rotated = (R @ points_centered.T).T

    # 获取归一化后的 X 和 Y (旋转后的轨迹)
    x_norm = points_rotated[:, 0]
    y_norm = points_rotated[:, 1]  # 垂直于理想路径的偏差

    # 2. 计算 AUC (使用梯形法则积分)
    # AUC 衡量轨迹与理想直线 (y=0) 之间的面积。我们只关心绝对偏差。
    # 我们只对从 0 到理想直线投影长度的区域积分

    # 确保 X 轴上的点是单调递增的，这在轨迹运动中通常满足。
    # 理想直线投影点 (x_norm) 必须是积分的 x 轴

    # 仅保留向前移动的轨迹点 (x_norm 必须是递增的)
    valid_indices = np.where(np.diff(x_norm, prepend=x_norm - 1) >= 0)
    x_norm_clean = x_norm[valid_indices]
    y_norm_abs = np.abs(y_norm[valid_indices])

    # 使用梯形法则计算 AUC
    auc_value = trapz(y_norm_abs, x_norm_clean)

    # 3. 速度剖面 (可选，但很有用)：平均瞬时速度
    # 速度是距离除以时间。由于原始代码没有记录时间戳，我们假定帧率恒定。
    # 平均速度：总路径长度 / RT
    path_length = np.sum(np.sqrt(np.diff(x_raw) ** 2 + np.diff(y_raw) ** 2))

    return auc_value, path_length  # 返回 AUC 和总路径长度 (作为平均速度的代理)


# --- 3. 数据处理与特征工程 ---

def feature_engineering(df):
    """对原始数据进行清洗，并计算所有高级特征。"""

    # 过滤掉超时试次
    df_clean = df[df['chose_delayed'] != 'timeout'].copy()

    # 转换为数值变量
    df_clean['chose_delayed_numeric'] = df_clean['chose_delayed'].astype(int)
    df_clean['comp_m'] = df_clean['comp_m'].astype(float)
    df_clean['participant'] = df_clean['participant'].astype('category')

    # 创建类别变量
    df_clean['sign_coded'] = df_clean['block_sign'].apply(lambda x: 1 if x == 'gain' else -1)
    df_clean['pressure_coded'] = df_clean['block_pressure'].apply(lambda x: 1 if x == 'high_pressure' else 0)

    # ----------------------------------------------------
    # 核心特征 1: 轨迹高级指标 (MD 已有，新增 AUC 和 Path Length)
    # ----------------------------------------------------

    auc_list =[]
    path_list =[]

    for _, row in df_clean.iterrows():
        x, y = parse_trajectory(row['raw_trajectory'])
        if len(x) > 0:
            auc, path_length = normalize_and_calculate_auc(x, y, row['choice_side'])
            auc_list.append(auc)
            path_list.append(path_length)
        else:
            auc_list.append(np.nan)
            path_list.append(np.nan)

    df_clean['auc'] = auc_list
    df_clean['path_length'] = path_list
    df_clean['mean_speed'] = df_clean['path_length'] / df_clean['rt']

    # ----------------------------------------------------
    # 核心特征 2: 计算主观价值差 (DDM / GLMM 的输入)
    # ----------------------------------------------------
    # 初始化 k 值为 0.05 (用于迭代，如果使用更复杂的 DDM，这里需要外部估计)

    # 暂时使用 comp_m 作为价值差的代理，直到 GLMM 提供了 IP 估计
    df_clean['value_diff_proxy'] = df_clean['comp_m'] - STANDARD_AMOUNT

    # 移除 NaN 值
    df_clean.dropna(subset=['auc', 'rt', 'max_deviation'], inplace=True)

    return df_clean


# --- 4. 统计分析：分层 k 值估计 (GLMM) ---

def estimate_hierarchical_k(df):
    """
    使用广义线性混合效应模型 (GLMM) 拟合心理测量函数，
    以鲁棒地估计 IP 和 k 值。
    P(Delayed) ~ 1 + comp_m + Sign * Pressure + (1 + comp_m | participant)
    """

    print("\n=====================================================")
    print("【4. GLMM 分层 k 值估计】")
    print("=====================================================")

    # 目标：通过 GLMM 找到 P(Delayed)=0.5 时的 comp_m (即 IP)
    # 我们使用 comp_m 作为连续预测变量，选择 (chose_delayed) 作为二元响应。
    # 允许截距和斜率在被试间变化 (随机效应)。

    # 注意：Logit 模型中的自变量 (comp_m) 必须是数值。
    # 我们将 Sign 和 Pressure 引入模型，检验它们是否影响 comp_m 的效应。

    # GLMM 公式：P(Delayed) ~ comp_m * Sign * Pressure + (1 | participant)
    # 随机效应 (1 | participant) 允许每个被试有不同的基线偏好 (截距)
    # 也可以包含 comp_m 的随机斜率：(1 + comp_m | participant)，但计算成本更高，这里简化。

    formula = "chose_delayed_numeric ~ comp_m * block_sign * block_pressure"

    # 使用 GLMM (广义线性混合模型) - family=Binomial for Logistic Regression
    # 随机效应只在截距上应用 (最常用且稳定的形式)

    try:
        model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
        print(model.summary())

        # 为了获得 IP，我们需要预测 P(Delayed) = 0.5。
        # Logit(P) = Beta_0 + Beta_1 * comp_m = 0 (当 P=0.5 时)
        # 此时 comp_m = -Beta_0 / Beta_1

        # 由于我们有交互项，IP 是条件依赖的。

        results = {}
        for sign in ['gain', 'loss']:
            for pressure in ['no_pressure', 'high_pressure']:

                # 构建该条件下的 Logit 方程系数 (Fixed Effects)
                intercept = model.params['Intercept']
                comp_m_coeff = model.params['comp_m']

                if sign == 'loss':
                    intercept += model.params.get('block_sign', 0)
                    comp_m_coeff += model.params.get('comp_m:block_sign', 0)

                if pressure == 'high_pressure':
                    intercept += model.params.get('block_pressure', 0)
                    comp_m_coeff += model.params.get('comp_m:block_pressure', 0)

                    # 包含三向交互项 (Sign * Pressure * comp_m)
                    if sign == 'loss':
                        # 查找 T.loss:T.high_pressure (或反向)
                        interaction_coeff = model.params.get('comp_m:block_sign:block_pressure', 0)
                        comp_m_coeff += interaction_coeff
                        # 查找 T.loss:T.high_pressure (非comp_m部分)
                        intercept += model.params.get('block_sign:block_pressure', 0)

                # 计算 IP
                ip = -intercept / comp_m_coeff if comp_m_coeff != 0 else np.nan

                # 计算 k 值
                k = (STANDARD_AMOUNT / ip - 1) / DAYS_DELAY if ip > 0 else np.nan

                results[f'{sign}_{pressure}'] = {'IP': ip, 'k_value': k}

        k_df = pd.DataFrame.from_dict(results, orient='index')
        print("\n--- 分层 k 值估计结果 (通过 GLMM 拟合的 IP) ---")
        print(k_df.to_string(float_format="%.4f"))
        return k_df

    except Exception as e:
        print(f"GLMM 拟合失败: {e}. 可能数据量不足或方差问题。")
        return pd.DataFrame()


# --- 5. 统计分析：过程指标 LME 分析 ---

def analyze_process_metrics(df):
    """
    使用线性混合效应模型 (LME) 分析 RT 和 MD/AUC 对实验操纵的响应。
    """

    print("\n=====================================================")
    print("【5. LME 过程指标分析 (RT, MD, AUC)】")
    print("=====================================================")

    # LME 模型允许我们将被试 (participant) 作为随机效应包含在内，
    # 从而正确处理重复测量设计。

    metrics = ['rt', 'max_deviation', 'auc', 'mean_speed']

    for metric in metrics:
        print(f"\n--- 正在拟合 {metric} 的 LME 模型 ---")

        # LME 公式：Metric ~ Sign * Pressure + (1 | participant)
        # 允许截距在被试间变化

        formula = f"{metric} ~ block_sign * block_pressure"

        try:
            # 拟合 LME 模型
            # 注意：如果数据是单被试，LME 会退化为 OLS/GLS，但对于多被试数据，它是必需的。
            model = smf.mixedlm(formula, data=df, groups=df['participant'],
                                re_formula="~1").fit(method='powell', maxiter=100)

            # 由于输出太长，我们只打印关键结果
            print(f"LME for {metric}: P-Values for Fixed Effects:")

            # 提取固定效应 P 值
            fixed_effects = model.pvalues[model.pvalues.index.str.contains('Intercept|block_sign|block_pressure')]

            print(fixed_effects.to_string(float_format="%.4f"))

            # 预测解释：
            if metric == 'max_deviation':
                print(f" - 预测结果: MD 越高，认知冲突越大。")
            elif metric == 'rt':
                print(f" - 预测结果: 压力应导致 RT 显著降低 (DDM 边界 a 降低)。")
            elif metric == 'auc':
                print(f" - 预测结果: AUC 是 MD 的替代品，反映冲突的持续性和深度。")

        except Exception as e:
            print(f"LME for {metric} 拟合失败: {e}")


# --- 6. DDM 整合的说明 (无需代码实现，只需阐述) ---

def ddm_integration_explanation():
    """解释如何将 MD/AUC 集成到 DDM 参数中。"""

    print("\n=====================================================")
    print("【6. DDM 整合的计算路线图】")
    print("=====================================================")

    print("为了完成神经计算分析，下一步是应用漂移扩散模型 (DDM)。")
    print("DDM 目标是将 RT 和选择的分布拟合到以下潜在参数：")

    print("| DDM 参数 | 理论意义 | 实验操纵映射 |")
    print("|---|---|---|")
    print("| v (漂移率) | 证据积累速度/效率 | 与 **主观价值差 (ΔV)** 成正比。ΔV 由 k 值决定。|")
    print("| a (边界分离) | 决策谨慎度/证据阈值 | **压力 (High Pressure)** 预期会降低 a 值 [2]。|")
    print("| t0 (非决策时间) | 感知和运动时间 | 几乎不随价值或压力变化。|")

    print("\nDDM 整合轨迹指标 (MD/AUC) 的策略：")
    print(
        "1. **MD/AUC 作为 v 的协变量:** 将 MD 或 AUC 作为试次级的协变量，纳入漂移率 v 的估计中 [3, 4]。例如，假设 $v = \beta_0 + \beta_1 \cdot \Delta V + \beta_2 \cdot MD$。这测试了几何冲突 (MD/AUC) 是否能独立解释证据积累速度 (v) 的变异性。")
    print(
        "2. **属性优先级 (Attribute Latency):** 分析规范化轨迹的早期阶段 (例如前 20%)，查看不同框架 (Gain/Loss) 下，鼠标是否更快地倾向于 '金额' 轴或 '时间' 轴 [5, 6]。")


# --- 7. 主执行块 ---

if __name__ == '__main__':

    if 'your_participant_id' in FILE_PATH:
        print("=====================================================")
        print("警告：请修改 FILE_PATH 变量，指向您实际的实验数据 CSV 文件。")
        print("=====================================================")
    else:
        # 1. 加载和预处理数据
        df_raw = pd.read_csv(FILE_PATH)
        df_processed = feature_engineering(df_raw)

        if df_processed.empty:
            print("处理后的数据为空，请检查原始 CSV 文件内容。")
        else:
            # 2. GLMM k 值估计
            k_results = estimate_hierarchical_k(df_processed)

            # 3. LME 过程指标分析
            analyze_process_metrics(df_processed)

            # 4. DDM 整合说明
            ddm_integration_explanation()

            print("\n=====================================================")
            print(
                "分析完成。请重点解读 GLMM 和 LME 模型的固定效应 (Fixed Effects) P 值，以验证您关于框架、压力和冲突的理论预测。")
            print("=====================================================")