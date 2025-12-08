"""
logistic_rt_interaction.py

说明：
- 将 ./data/ 下所有 CSV 合并（每个 CSV 包含 participant 的多次试次）
- 自动识别列：comp_m (Amount), rt (reaction time), chose_delayed (binary choice),
  participant (subject), block_pressure (condition)
- NOTE: Delay (7 days) 是常数 — 因为固定不变，无法用于回归估计敏感度；
  本脚本不把“延迟”当作解释变量，而把 rt 视为反应时间作为协变量。
- 输出：pooled logistic, per-condition logistic, GEE, 并做 RT × Condition 交互检验
- 直接在控制台显示结果（不保存文件）
"""
import os, glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
STANDARDIZE = True     # 将 Amount 和 RT 标准化（推荐）
MIN_N_PER_COND = 8     # 每条件最小样本量以进行按条件拟合

# expected candidates
EXPECTED = {
    'amount': ['comp_m', 'amount', 'comp'],
    'rt': ['rt', 'reaction', 'reaction_time', 'response_time'],
    'choice': ['chose_delayed', 'choice', 'chose', 'resp'],
    'subject': ['participant', 'subject', 'subj', 'pid'],
    'condition': ['block_pressure', 'pressure', 'cond', 'condition', 'block']
}

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # substring match
    for cand in candidates:
        for lc, orig in lower.items():
            if cand.lower() in lc:
                return orig
    return None

def auto_detect(df):
    cols = list(df.columns)
    return {
        'amount': find_col(cols, EXPECTED['amount']),
        'rt': find_col(cols, EXPECTED['rt']),
        'choice': find_col(cols, EXPECTED['choice']),
        'subject': find_col(cols, EXPECTED['subject']),
        'condition': find_col(cols, EXPECTED['condition'])
    }

def prepare(df, detected, standardize=True):
    d = df.copy()
    # choice -> 0/1
    chcol = detected['choice']
    if chcol is None:
        raise RuntimeError("未识别到 choice 列，请确保包含 'chose_delayed' 或类似列。")
    ch = d[chcol]
    if pd.api.types.is_bool_dtype(ch):
        d['_Choice'] = ch.astype(int)
    elif pd.api.types.is_numeric_dtype(ch):
        uniq = sorted(ch.dropna().unique())
        if set(uniq) <= {0,1}:
            d['_Choice'] = ch.astype(int)
        elif len(uniq) == 2:
            mapping = {uniq[0]:0, uniq[1]:1}
            d['_Choice'] = ch.map(mapping)
        else:
            d['_Choice'] = (ch > ch.median()).astype(int)
    else:
        vals = ch.astype(str)
        top = vals.value_counts().index.tolist()
        if len(top) >= 2:
            mapping = {top[0]:1, top[1]:0}   # mode -> 1
            d['_Choice'] = vals.map(lambda x: mapping.get(x, np.nan))
        else:
            d['_Choice'] = vals.map(lambda x: 1 if x==top[0] else np.nan)

    # amount and rt numeric
    amtcol = detected['amount']
    rtcol = detected['rt']
    if amtcol is None:
        raise RuntimeError("未识别到金额列（comp_m）。")
    if rtcol is None:
        raise RuntimeError("未识别到反应时间列（rt）。")

    d['_Amount'] = pd.to_numeric(d[amtcol], errors='coerce')
    d['_RT'] = pd.to_numeric(d[rtcol], errors='coerce')

    # subject & condition
    subjcol = detected['subject']
    condcol = detected['condition']
    d['_Subject'] = d[subjcol].astype(str) if (subjcol and subjcol in d.columns) else d.index.astype(str)
    d['_Cond'] = d[condcol].astype(str).fillna('NA') if (condcol and condcol in d.columns) else 'NA'

    # drop missing
    d = d.dropna(subset=['_Choice','_Amount','_RT']).copy()

    # standardize if requested
    if standardize:
        scaler = StandardScaler()
        d[['_Amount_z','_RT_z']] = scaler.fit_transform(d[['_Amount','_RT']])
        d['_Amount_model'] = d['_Amount_z']
        d['_RT_model'] = d['_RT_z']
    else:
        d['_Amount_model'] = d['_Amount']
        d['_RT_model'] = d['_RT']

    return d

def fit_logit(formula, df):
    return smf.logit(formula=formula, data=df).fit(disp=False, maxiter=200)

def fit_gee(formula, df, group='_Subject'):
    return GEE.from_formula(formula, groups=group, data=df, family=Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()

def summarize(result, name):
    if result is None:
        print(f"{name}: No result")
        return
    if isinstance(result, Exception):
        print(f"{name}: error -> {result}")
        return
    print(f"\n=== {name} ===")
    try:
        print(result.summary())
    except Exception:
        pass
    params = result.params
    alpha = params.get('Intercept', params.iloc[0])
    b_amt = params.get('_Amount_model', np.nan)
    b_rt = params.get('_RT_model', np.nan)
    print(f"\n{name} coefficients (log-odds): alpha={alpha:.6f}, beta_amount={b_amt:.6f}, beta_RT={b_rt:.6f}")
    try:
        conf = result.conf_int()
        def ci(term): 
            if term in conf.index:
                return conf.loc[term].values
            return (np.nan, np.nan)
        ca = ci('_Amount_model'); cr = ci('_RT_model'); ci_a = ci('Intercept')
        print(f"  95% CI intercept={ci_a}, amount={ca}, RT={cr}")
        print(f"  OR_amount={np.exp(b_amt):.4f}, OR_RT={np.exp(b_rt):.4f}")
    except Exception:
        pass
    try:
        pseudo = 1 - result.llf / result.llnull
        print(f"  McFadden pseudo-R^2 = {pseudo:.4f}")
    except Exception:
        pass

def interpret_interaction(result, result_name, factor_name='_RT_model'):
    if result is None or isinstance(result, Exception):
        print(f"{result_name}: no result to interpret")
        return
    params = result.params
    pvals = getattr(result, "pvalues", None)
    # find interaction term name containing RT_model and no_pressure (or second category)
    inter_terms = [t for t in params.index if (factor_name in t and ('no_pressure' in t or 'No_pressure' in t or 'noPressure' in t))]
    if not inter_terms:
        inter_terms = [t for t in params.index if (factor_name in t and ':' in t and '_Cond' in t)]
    if not inter_terms:
        print(f"{result_name}: interaction term not found.")
        return
    t = inter_terms[0]
    coef = params[t]
    p = pvals[t] if (pvals is not None and t in pvals.index) else None
    print(f"\n{result_name} interaction term '{t}': coef={coef:.4f}, p={p}")
    if p is not None:
        if p < 0.05:
            if coef > 0:
                print("  结论：交互为正且显著 → no_pressure 下 RT 的效应比 high_pressure 大 → 支持 high_pressure 下 RT 敏感度减弱（属性窄化）。")
            else:
                print("  结论：交互为负且显著 → no_pressure 下 RT 的效应比 high_pressure 小 → 与属性窄化相反。")
        else:
            print("  结论：交互不显著 → 无证据显示压力改变了对 RT 的敏感度。")
    else:
        print("  无 p 值；请检查 summary() 输出。")

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        print("data 文件夹下未找到 CSV 文件。请把 CSV 放入 ./data/ 然后重试。")
        return
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['_src'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"读取 {f} 失败：{e}")
    full = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"已加载 {len(files)} 个 CSV，合并后 shape = {full.shape}")
    print("列名列表：", list(full.columns))

    detected = auto_detect(full)
    print("\n自动识别列（None 表示未找到）:")
    for k,v in detected.items():
        print(f"  {k}: {v}")

    # Prepare
    work = prepare(full, detected, standardize=STANDARDIZE)
    print("\n注意：延迟（delay）为实验固定值 7 天，不作为回归变量（常数）。")
    print("\n各条件样本量：")
    print(work['_Cond'].value_counts().to_string())

    # set categorical order: prefer 'high_pressure' as baseline if present
    uniq = list(work['_Cond'].unique())
    if 'high_pressure' in uniq:
        cats = ['high_pressure'] + [c for c in uniq if c != 'high_pressure']
    else:
        cats = uniq
    work['_Cond_cat'] = pd.Categorical(work['_Cond'], categories=cats)

    # pooled logistic (Amount + RT)
    pooled = None
    try:
        pooled = fit_logit("_Choice ~ _Amount_model + _RT_model", work)
        summarize(pooled, "Pooled Logistic (Amount + RT)")
    except Exception as e:
        print("Pooled logistic 拟合失败：", e)

    # per-condition
    cond_models = {}
    for cond, sub in work.groupby('_Cond'):
        if len(sub) < MIN_N_PER_COND:
            cond_models[cond] = None
            print(f"Condition {cond}: N={len(sub)} < {MIN_N_PER_COND}，跳过按条件拟合。")
            continue
        try:
            m = fit_logit("_Choice ~ _Amount_model + _RT_model", sub)
            cond_models[cond] = m
            summarize(m, f"Logit (cond={cond})")
        except Exception as e:
            cond_models[cond] = e
            print(f"Condition {cond} 拟合失败：", e)

    # GEE (population-averaged) main effects
    gee = None
    try:
        gee = fit_gee("_Choice ~ _Amount_model + _RT_model + _Cond_cat", work, group='_Subject')
        summarize(gee, "GEE by Subject (main effects)")
    except Exception as e:
        print("GEE 拟合失败：", e)

    # Interaction tests: RT × Condition
    print("\n=== Interaction test: RT × Condition ===")
    pooled_inter = None; gee_inter = None
    try:
        pooled_inter = fit_logit("_Choice ~ _Amount_model + _RT_model * _Cond_cat", work)
        summarize(pooled_inter, "Pooled Logistic (with RT × Condition)")
        interpret_interaction(pooled_inter, "Pooled_interaction")
    except Exception as e:
        print("Pooled interaction 拟合失败：", e)
    try:
        gee_inter = fit_gee("_Choice ~ _Amount_model + _RT_model * _Cond_cat", work, group='_Subject')
        summarize(gee_inter, "GEE (with RT × Condition)")
        interpret_interaction(gee_inter, "GEE_interaction")
    except Exception as e:
        print("GEE interaction 拟合失败：", e)

    print("\n注意与结论提示：")
    print(" - Delay=7天为常数，无法纳入回归作为解释变量。")
    print(" - 若你想检验压力是否改变对延迟（delay）的敏感度，需设计不同延迟值的试次（delay must vary）。")
    print(" - 当前脚本检验的是压力是否改变对 Reaction Time (rt) 的敏感度（RT × Condition 交互）。")

if __name__ == '__main__':
    main()
