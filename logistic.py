#!/usr/bin/env python3
# attribute_sensitivity_auto.py (modified: add interaction checks)
# 直接运行即可：将所有数据 csv 放到 ./data/ 下，脚本会自动读取并显示结果与图（不保存文件）。

import os
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ------------------------
# 配置（如需改为标准化可把下面设为 True）
# ------------------------
DATA_DIR = 'data'
STANDARDIZE = True  # 是否对 Amount / Time 做 z-score 标准化
MIN_N_PER_COND = 10  # 每条件最小样本量，否则跳过按条件拟合

# ------------------------
# 列自动匹配规则（优先级匹配）
# ------------------------
EXPECTED_NAMES = {
    'amount': ['comp_m', 'thisrow.t', 'amount', 'amt', 'comp'],
    'time': ['rt', 'time', 'delay', 'latency', 'thisrow.t'],
    'choice': ['chose_delayed', 'choice', 'chose', 'resp', 'response'],
    'subject': ['participant', 'subject', 'subj', 'pid', 'participant_id'],
    'condition': ['block_pressure', 'pressure', 'cond', 'condition', 'block']
}

def find_first_match(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # substring match
    for cand in candidates:
        for lower, orig in cols_lower.items():
            if cand.lower() in lower:
                return orig
    return None

def auto_detect_cols(df):
    cols = list(df.columns)
    detected = {}
    detected['choice'] = find_first_match(cols, EXPECTED_NAMES['choice'])
    detected['amount'] = find_first_match(cols, EXPECTED_NAMES['amount'])
    detected['time'] = find_first_match(cols, EXPECTED_NAMES['time'])
    detected['subject'] = find_first_match(cols, EXPECTED_NAMES['subject'])
    detected['condition'] = find_first_match(cols, EXPECTED_NAMES['condition'])
    return detected

# ------------------------
# 数据准备
# ------------------------
def prepare_df(df, detected, standardize=False):
    df2 = df.copy()
    # Choice -> 0/1
    if detected['choice'] is None:
        raise ValueError("未能自动识别 Choice 列，请确认 CSV 列头中包含 'chose_delayed' 或类似名称。")
    ch = df2[detected['choice']]
    if pd.api.types.is_bool_dtype(ch):
        df2['_Choice'] = ch.astype(int)
    elif pd.api.types.is_numeric_dtype(ch):
        uniq = sorted(ch.dropna().unique())
        if set(uniq) <= {0,1}:
            df2['_Choice'] = ch.astype(int)
        elif len(uniq) == 2:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            df2['_Choice'] = ch.map(mapping)
        else:
            df2['_Choice'] = (ch > ch.median()).astype(int)
    else:
        vals = ch.astype(str)
        top = vals.value_counts().index.tolist()
        if len(top) >= 2:
            mapping = {top[0]: 1, top[1]: 0}  # 众数映射为 1
            df2['_Choice'] = vals.map(lambda x: mapping.get(x, np.nan))
        else:
            df2['_Choice'] = vals.map(lambda x: 1 if x == top[0] else np.nan)

    # Amount / Time -> numeric
    amt_col = detected['amount']
    time_col = detected['time']
    if amt_col is None and time_col is None:
        raise ValueError("未识别到 Amount 或 Time 列（期望 comp_m / thisRow.t / rt 等）。")
    df2['_Amount'] = pd.to_numeric(df2[amt_col], errors='coerce') if amt_col is not None else np.nan
    df2['_Time'] = pd.to_numeric(df2[time_col], errors='coerce') if time_col is not None else np.nan

    # subject / condition
    subj_col = detected['subject']
    cond_col = detected['condition']
    df2['_Subject'] = df2[subj_col].astype(str) if (subj_col is not None and subj_col in df2.columns) else df2.index.astype(str)
    df2['_Cond'] = df2[cond_col].astype(str).fillna('NA') if (cond_col is not None and cond_col in df2.columns) else 'NA'

    # drop NA
    df2 = df2.dropna(subset=['_Choice', '_Amount', '_Time']).copy()

    # standardize optional
    if standardize:
        scaler = StandardScaler()
        df2[['_Amount_z', '_Time_z']] = scaler.fit_transform(df2[['_Amount', '_Time']])
        df2['_Amount_model'] = df2['_Amount_z']
        df2['_Time_model'] = df2['_Time_z']
    else:
        df2['_Amount_model'] = df2['_Amount']
        df2['_Time_model'] = df2['_Time']

    return df2

# ------------------------
# 模型拟合与输出
# ------------------------
def fit_pooled_logit(df):
    formula = "_Choice ~ _Amount_model + _Time_model"
    model = smf.logit(formula=formula, data=df).fit(disp=False, maxiter=200)
    return model

def fit_logit_by_condition(df):
    res = {}
    for cond, sub in df.groupby('_Cond'):
        if sub.shape[0] < MIN_N_PER_COND:
            res[cond] = None
            continue
        try:
            res[cond] = smf.logit(formula="_Choice ~ _Amount_model + _Time_model", data=sub).fit(disp=False, maxiter=200)
        except Exception as e:
            res[cond] = e
    return res

def fit_gee_by_subject(df):
    df['_Cond_cat'] = df['_Cond'].astype('category')
    formula = "_Choice ~ _Amount_model + _Time_model + _Cond_cat"
    model = GEE.from_formula(formula, groups="_Subject", data=df, family=Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    return res

def summarize_and_print(result, name):
    if result is None:
        print(f"\n{name}: 无可用结果（None）")
        return
    if isinstance(result, Exception):
        print(f"\n{name}: 拟合出错 -> {result}")
        return
    print(f"\n=== {name} summary ===")
    try:
        print(result.summary())
    except Exception:
        print("无法打印 summary()，但是会打印主要系数：")
    # 报告 alpha / beta
    params = result.params
    alpha = params.get('Intercept', params.index[0] and params.iloc[0])
    beta_a = params.get('_Amount_model', np.nan)
    beta_t = params.get('_Time_model', np.nan)
    print(f"\n{name} coefficients (log-odds):")
    print(f"  alpha (Intercept) = {float(alpha):.6f}")
    print(f"  beta_amount (_Amount_model) = {float(beta_a):.6f}")
    print(f"  beta_time   (_Time_model) = {float(beta_t):.6f}")
    # 若有置信区间与 OR
    try:
        conf = result.conf_int()
        ci_a = conf.loc['Intercept'].values if 'Intercept' in conf.index else conf.iloc[0].values
        ci_amt = conf.loc['_Amount_model'].values if '_Amount_model' in conf.index else [np.nan, np.nan]
        ci_time = conf.loc['_Time_model'].values if '_Time_model' in conf.index else [np.nan, np.nan]
        or_a = np.exp(alpha)
        or_amt = np.exp(beta_a)
        or_time = np.exp(beta_t)
        print("\n  95% CI (log-odds):")
        print(f"    alpha CI = ({ci_a[0]:.4f}, {ci_a[1]:.4f})")
        print(f"    amount CI = ({ci_amt[0]:.4f}, {ci_amt[1]:.4f})")
        print(f"    time CI = ({ci_time[0]:.4f}, {ci_time[1]:.4f})")
        print("\n  Odds ratios (OR):")
        print(f"    OR_alpha = {or_a:.4f}")
        print(f"    OR_amount = {or_amt:.4f}")
        print(f"    OR_time = {or_time:.4f}")
    except Exception:
        pass
    # McFadden pseudo R2 if available
    try:
        pseudo = 1 - result.llf / result.llnull
        print(f"\n  McFadden pseudo-R² = {pseudo:.4f}")
    except Exception:
        pass

# ------------------------
# Interaction detection helper
# ------------------------
def interpret_time_interaction(result, result_name):
    """
    尝试找到包含 '_Time_model' 且包含 'no_pressure' 的交互项名并自动解读。
    若找不到交互项，会在控制台说明。
    """
    if result is None or isinstance(result, Exception):
        print(f"\n{result_name}: 无结果可解读交互项。")
        return
    params = result.params
    pvals = None
    try:
        pvals = result.pvalues
    except Exception:
        pvals = None

    # 在参数名中寻找交互项（不依赖确切分隔符）
    inter_terms = [t for t in params.index if ('_Time_model' in t and ('no_pressure' in t or 'No_pressure' in t or 'noPressure' in t))]
    # 如果没找到，也尝试找 _Time_model: 之后的任何 Cond 类别差异项
    if not inter_terms:
        inter_terms = [t for t in params.index if ('_Time_model' in t and ':' in t and '_Cond' in t)]
    if not inter_terms:
        print(f"\n{result_name}: 未找到显式的 Time×Condition 交互项（参数名列表中没有包含 '_Time_model' 与 condition 的项）。")
        return

    tname = inter_terms[0]
    coef = params[tname]
    p = pvals[tname] if pvals is not None and tname in pvals.index else None
    print(f"\n{result_name} found interaction term '{tname}': coef = {coef:.4f}, p = {p}")
    if p is not None:
        if p < 0.05:
            if coef > 0:
                print("  解释：交互项显著且为正 → 在 no_pressure 下 Time 的效应比在 high_pressure 下更强 ⇒ 支持 “high_pressure 下 Time 敏感度减弱（属性窄化）”。")
            else:
                print("  解释：交互项显著且为负 → 在 no_pressure 下 Time 的效应比在 high_pressure 下更弱 ⇒ 与属性窄化相反。")
        else:
            print("  解释：交互项不显著（p >= 0.05）→ 无证据表明压力改变了 Time 的敏感度（不能支持属性窄化）。")
    else:
        print("  无 p 值可用；请手动检查 summary() 中该项的置信区间与显著性。")

# ------------------------
# 入口
# ------------------------
def main():
    # 查找 CSV
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if len(files) == 0:
        print(f"目录 {DATA_DIR} 中未找到 CSV 文件，请把 CSV 放到该目录后重试。")
        return

    # 读取并合并
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d['_source_file'] = os.path.basename(f)
            dfs.append(d)
        except Exception as e:
            print(f"读取文件 {f} 失败：{e}")
    full = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"已加载 {len(files)} 个 CSV，合并后 shape = {full.shape}")
    print("列名列表：", list(full.columns))

    # 自动检测列
    detected = auto_detect_cols(full)
    print("\n自动识别的列（None 表示未找到）：")
    for k, v in detected.items():
        print(f"  {k}: {v}")

    # 准备数据
    try:
        work = prepare_df(full, detected, standardize=STANDARDIZE)
    except Exception as e:
        print("准备数据失败：", e)
        return

    print("\n各条件样本量：")
    print(work['_Cond'].value_counts().to_string())

    # 拟合 pooled logistic
    pooled = None
    try:
        pooled = fit_pooled_logit(work)
        summarize_and_print(pooled, "Pooled Logistic")
    except Exception as e:
        print("Pooled logistic 拟合失败：", e)

    # 按 condition 拟合
    cond_models = fit_logit_by_condition(work)
    for cond, res in cond_models.items():
        if res is None:
            print(f"\nCondition '{cond}': 样本量 < {MIN_N_PER_COND} 或已跳过。")
        elif isinstance(res, Exception):
            print(f"\nCondition '{cond}': 拟合异常 -> {res}")
        else:
            summarize_and_print(res, f"Logit (cond={cond})")

    # GEE by subject (main effects)
    gee = None
    try:
        gee = fit_gee_by_subject(work)
        summarize_and_print(gee, "GEE by Subject")
    except Exception as e:
        print("GEE 拟合失败：", e)

    # --------------------------
    # 交互检测（Time × Condition）
    # --------------------------
    # 先把 Condition 设为 category，并尽量把 'high_pressure' 放在第一个（reference）
    try:
        unique_conds = list(work['_Cond'].astype(str).unique())
        if 'high_pressure' in unique_conds:
            cats = ['high_pressure'] + [c for c in unique_conds if c != 'high_pressure']
        else:
            cats = unique_conds
        work['_Cond_cat'] = pd.Categorical(work['_Cond'], categories=cats)
    except Exception:
        work['_Cond_cat'] = work['_Cond'].astype('category')

    print("\n=== Interaction tests: Time × Condition ===")
    # pooled interaction
    try:
        pooled_inter = smf.logit(formula = "_Choice ~ _Amount_model + _Time_model * _Cond_cat", data=work).fit(disp=False, maxiter=200)
        print("\n--- Pooled logistic with interaction ---")
        print(pooled_inter.summary())
        interpret_time_interaction(pooled_inter, "Pooled_interaction")
    except Exception as e:
        print("Pooled interaction 拟合失败：", e)
        pooled_inter = None

    # GEE interaction
    try:
        gee_inter = GEE.from_formula("_Choice ~ _Amount_model + _Time_model * _Cond_cat",
                                    groups="_Subject", data=work, family=Binomial(),
                                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        print("\n--- GEE with interaction ---")
        print(gee_inter.summary())
        interpret_time_interaction(gee_inter, "GEE_interaction")
    except Exception as e:
        print("GEE interaction 拟合失败：", e)
        gee_inter = None

    print("\n全部分析完成（在控制台显示输出，未保存任何文件）。")

if __name__ == "__main__":
    main()
