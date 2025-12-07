# dg_ddm_correct_parse.py
"""
Fix parsing of chose_delayed and robust DG-DDM refit pipeline.

Usage:
    python dg_ddm_correct_parse.py

Outputs:
 - figs/ (PNG diagnostic plots)
 - dg_ddm_corrected_results.json
"""
import os, json, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------- config ----------
DATA_PATH = "data/416555_Mouse_Track_Study_2025-12-05_20h54.21.935.csv"
OUT_JSON = "dg_ddm_corrected_results.json"
FIG_DIR = "figs"
N_BINS = 8
MULTI_STARTS = 30
MIN_TRIALS_TO_FIT = 6
EPS_V = 1e-2
np.random.seed(2)
os.makedirs(FIG_DIR, exist_ok=True)

# -------- helpers ----------
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def save_fig(plt_obj, name):
    path = os.path.join(FIG_DIR, name)
    plt_obj.tight_layout()
    plt_obj.savefig(path, dpi=150)
    print("Saved:", path)
    plt_obj.clf()

# -------- robust column detection (fixed) ----------
def detect_columns_strict(df):
    cols = list(df.columns)
    # Choose column by clearer rules:
    amt_cols = [c for c in cols if any(k in c.lower() for k in ("comp_m","amount","amt","value"))]
    # Strict choice detection: 'chose', 'choice' but not words containing 'delay' as main word
    choice_cols = [c for c in cols if ("chose" in c.lower() or "choice" in c.lower() or c.lower().startswith("resp") )]
    # RT detection
    rt_cols = [c for c in cols if c.lower()=="rt" or "reaction" in c.lower() or "response_time" in c.lower()]
    # Pressure
    pressure_cols = [c for c in cols if "press" in c.lower() or "pressure" in c.lower()]
    # Time - prefer explicit 'thisRow.t' or 't'
    time_cols = [c for c in cols if c.lower() in ("thisrow.t","t","time","trial","trial_index")]
    # Delay strict: only columns containing 'delay' or starting with 'd_' or containing 'delay_' but exclude 'chose_delayed'
    delay_cols = [c for c in cols if (("delay" in c.lower() or c.lower().startswith("d_")) and "chose" not in c.lower())]
    return amt_cols, choice_cols, rt_cols, pressure_cols, time_cols, delay_cols

# --------- preprocessing (correct) ----------
def preprocess(path):
    df = pd.read_csv(path)
    amt_cols, choice_cols, rt_cols, pressure_cols, time_cols, delay_cols = detect_columns_strict(df)
    print("Detected:")
    print(" amt_cols:", amt_cols)
    print(" choice_cols:", choice_cols)
    print(" rt_cols:", rt_cols)
    print(" pressure_cols:", pressure_cols)
    print(" time_cols:", time_cols)
    print(" delay_cols (strict):", delay_cols)

    if len(amt_cols)==0:
        raise RuntimeError("No amount-like column found (comp_m expected).")
    amt_col = amt_cols[0]
    # choice col: prefer 'chose_delayed' if present in choice_cols list
    if len(choice_cols)==0:
        raise RuntimeError("No choice-like column found.")
    # If multiple candidates, choose one that contains 'chose'
    chosen_choice = None
    for c in choice_cols:
        if "chose" in c.lower():
            chosen_choice = c; break
    if chosen_choice is None:
        chosen_choice = choice_cols[0]
    # pick rt, pressure, time
    rt_col = rt_cols[0] if len(rt_cols)>0 else None
    pressure_col = pressure_cols[0] if len(pressure_cols)>0 else None
    time_col = time_cols[0] if len(time_cols)>0 else None
    delay_col = delay_cols[0] if len(delay_cols)>0 else None

    work = df.copy()
    # amount
    work['comp_m'] = pd.to_numeric(work[amt_col], errors='coerce')

    # choice parsing: explicit handling of 'True'/'False' and strings
    s = work[chosen_choice].astype(str).str.lower()
    # treat 'true' ->1, 'false'->0; remove 'timeout' rows
    mask_timeout = s == 'timeout'
    if mask_timeout.any():
        print("Dropping", mask_timeout.sum(), "timeout rows from choice column.")
        work = work[~mask_timeout].copy()
        s = work[chosen_choice].astype(str).str.lower()

    mapping = {'true':1.0, 'false':0.0, '1':1.0, '0':0.0}
    # numeric convert fallback
    def map_choice(val):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        if v in mapping: return mapping[v]
        try:
            return float(v)
        except:
            return np.nan
    work['choice'] = work[chosen_choice].apply(map_choice)

    # rt
    if rt_col is not None:
        work['rt'] = pd.to_numeric(work[rt_col], errors='coerce')
        work.loc[work['rt']<=0,'rt'] = np.nan
    else:
        work['rt'] = np.nan

    # pressure
    if pressure_col is not None:
        work['block_pressure'] = work[pressure_col].astype(str)
    else:
        work['block_pressure'] = 'no_pressure'

    # time handling
    if time_col is not None:
        work['t'] = pd.to_numeric(work[time_col], errors='coerce')
        if work['t'].isna().all():
            print("time column exists but all NaN -> fallback to trial index")
            if 'participant' in work.columns:
                work['t'] = work.groupby(['participant','block_pressure']).cumcount()
            else:
                work['t'] = np.arange(len(work))
    else:
        if 'participant' in work.columns:
            work['t'] = work.groupby(['participant','block_pressure']).cumcount()
        else:
            work['t'] = np.arange(len(work))

    # delay (strict)
    if delay_col is not None:
        work['delay_attr'] = pd.to_numeric(work[delay_col], errors='coerce')
    else:
        work['delay_attr'] = np.nan

    # drop rows with missing comp_m or choice
    before = len(work)
    work = work[~work['comp_m'].isna() & ~work['choice'].isna()].copy()
    print(f"Dropped {before-len(work)} rows; remaining {len(work)} rows.")

    # final diagnostics
    print("Choice unique values:", np.unique(work['choice'].astype(str))[:10])
    print("Delay attr NaN count:", work['delay_attr'].isna().sum(), "/", len(work))

    return work

# -------- modelling functions ----------
def w_amt_t(t, alpha, lam, c): return alpha * np.exp(-lam * t) + c
def w_time_t(t, alpha, t0, tau, c): return alpha / (1.0 + np.exp(-(t - t0)/(tau+1e-12))) + c

# simplified model when delay absent: params = [alpha, lam, c, bias, k_choice]
def predict_v_simplified(params, t, A):
    alpha, lam, c, bias, k_choice = params
    w1 = w_amt_t(t, alpha, lam, c)
    v = w1 * A + bias
    return v, w1

def loss_simplified(params, t, A, Y, reg=1e-3):
    v,_ = predict_v_simplified(params, t, A)
    if np.isnan(v).any() or not np.isfinite(v).all():
        return 1e8
    logits = params[4] * v
    logits = np.clip(logits, -100, 100)
    p = sigmoid(logits)
    eps=1e-9
    nll = -np.sum(Y*np.log(p+eps) + (1-Y)*np.log(1-p+eps))
    l2 = reg*np.sum(np.square(params[:4]))
    return nll + l2

# full model if delay present: params = [alpha_amt, lam_amt, c_amt, alpha_time, t0, tau, c_time, bias, k_choice]
def predict_v_full(params, t, A, D):
    alpha_a, lam_a, c_a, alpha_t, t0, tau, c_t, bias, k_choice = params
    w1 = w_amt_t(t, alpha_a, lam_a, c_a)
    w2 = w_time_t(t, alpha_t, t0, tau, c_t)
    v = w1*A + w2*D + bias
    return v, w1, w2

def loss_full(params, t, A, D, Y, reg=1e-3):
    v,_,_ = predict_v_full(params, t, A, D)
    if np.isnan(v).any() or not np.isfinite(v).all():
        return 1e8
    logits = params[8] * v
    logits = np.clip(logits, -100, 100)
    p = sigmoid(logits)
    eps=1e-9
    nll = -np.sum(Y*np.log(p+eps) + (1-Y)*np.log(1-p+eps))
    l2 = reg*np.sum(np.square(params[:7]))  # regularize weights
    return nll + l2

# multi-start wrappers
def fit_simplified_multistart(t,A,Y, nstarts=MULTI_STARTS):
    best=None; best_loss=1e20
    # quick logistic fit to seed bias/slope
    try:
        def logit_nll(b):
            logits = b[0] + b[1]*A
            p = sigmoid(logits)
            return -np.sum(Y*np.log(p+1e-9)+(1-Y)*np.log(1-p+1e-9))
        res0 = minimize(logit_nll, x0=np.array([0.0,1.0]), method='BFGS')
        b0, b1 = (res0.x[0], res0.x[1]) if res0.success else (0.0,1.0)
    except Exception:
        b0,b1 = 0.0, 1.0
    for i in range(nstarts):
        init = np.array([ np.random.uniform(0.2,2.0), np.random.uniform(1e-4,0.5),
                          np.random.uniform(-0.5,0.5), b0 + np.random.normal(0,0.5), max(0.1,abs(b1)+np.random.normal(0,0.5)) ])
        bounds = [(1e-6,10),(1e-6,2),(-5,5),(-5,5),(1e-4,100)]
        res = minimize(lambda p: loss_simplified(p,t,A,Y), init, method='L-BFGS-B', bounds=bounds)
        if res.fun < best_loss:
            best_loss = res.fun; best = res.x
    return best, best_loss

def fit_full_multistart(t,A,D,Y, nstarts=MULTI_STARTS):
    best=None; best_loss=1e20
    for i in range(nstarts):
        init = np.array([
            np.random.uniform(0.2,2.0), np.random.uniform(1e-4,0.5), np.random.uniform(-0.5,0.5),
            np.random.uniform(-1,1), np.median(t), np.random.uniform(1.0,10.0), np.random.uniform(-0.5,0.5),
            np.random.uniform(-1,1), np.random.uniform(0.5,5.0)
        ])
        bounds = [(1e-6,10),(1e-6,2),(-5,5),(-5,5),(np.min(t)-50,np.max(t)+50),(1e-3,100),(-5,5),(-5,5),(1e-3,20)]
        res = minimize(lambda p: loss_full(p,t,A,D,Y), init, method='L-BFGS-B', bounds=bounds)
        if res.fun < best_loss:
            best_loss = res.fun; best = res.x
    return best, best_loss

# RT linear fit given v: RT ~ ndt + k_rt / (|v|+eps)
def fit_rt_given_v(v, RT):
    mask = ~np.isnan(RT) & np.isfinite(RT)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = 1.0/(np.abs(v)+EPS_V)
    X = np.vstack([np.ones(mask.sum()), x[mask]]).T
    y = RT[mask]
    coef, *_ = np.linalg.lstsq(X,y,rcond=None)
    return float(coef[0]), float(coef[1])

# plotting
def plot_obs_pred_binned(t, Y, p_hat, scope_name, fname):
    bins = np.linspace(np.min(t), np.max(t), N_BINS+1)
    idx = np.digitize(t, bins)-1
    idx[idx<0]=0; idx[idx>=N_BINS]=N_BINS-1
    centers = 0.5*(bins[:-1]+bins[1:])
    obs = [np.nan if not np.any(idx==i) else np.nanmean(Y[idx==i]) for i in range(N_BINS)]
    pred= [np.nan if not np.any(idx==i) else np.nanmean(p_hat[idx==i]) for i in range(N_BINS)]
    plt.figure(figsize=(7,4))
    plt.plot(centers, obs, '-o', label='observed')
    plt.plot(centers, pred, '-o', label='predicted')
    plt.ylim(-0.05,1.05); plt.xlabel('t'); plt.ylabel('P(choice delayed)')
    plt.title(f"{scope_name}: Observed vs Predicted (binned)")
    plt.legend()
    save_fig(plt, fname)

def plot_weight_curve(t, w_func_vals, label, scope_name, fname):
    plt.figure(figsize=(7,3))
    plt.plot(t, w_func_vals, label=label)
    plt.title(f"{scope_name}: {label}")
    plt.xlabel('t'); plt.legend()
    save_fig(plt, fname)

# -------------- pipeline --------------
def main():
    data = preprocess(DATA_PATH)

    scopes = [("all", data)]
    for blk in data['block_pressure'].unique():
        scopes.append((str(blk), data[data['block_pressure']==blk].copy()))

    out = {}
    for scope_name, df in scopes:
        print("\n=== scope:", scope_name, "n=", len(df))
        if len(df) < MIN_TRIALS_TO_FIT:
            print("too few trials, skip")
            out[scope_name] = {"fitted": False, "reason":"too_few"}
            continue

        t = pd.to_numeric(df['t'], errors='coerce').values.astype(float)
        A = pd.to_numeric(df['comp_m'], errors='coerce').values.astype(float)
        D = pd.to_numeric(df['delay_attr'], errors='coerce').values.astype(float)
        Y = pd.to_numeric(df['choice'], errors='coerce').values.astype(float)
        RT = pd.to_numeric(df['rt'], errors='coerce').values.astype(float)

        # standardize A and D
        A_mean = np.nanmean(A) if not np.isnan(np.nanmean(A)) else 0.0
        A_std = (np.nanstd(A)+1e-8) if (not np.isnan(np.nanstd(A)) and np.nanstd(A)>0) else 1.0
        A_z = (A - A_mean)/A_std
        if np.isnan(D).all():
            D_present = False
        else:
            D_filled = np.nan_to_num(D, nan=0.0)
            D_mean = np.nanmean(D_filled); D_std = np.nanstd(D_filled)+1e-8
            D_z = (D_filled - D_mean)/D_std
            D_present = True

        # t fallback
        if np.isnan(t).all():
            t0 = np.arange(len(t))
        else:
            t0 = t - np.nanmin(t)
            if np.nanmax(t0) > 1000:
                t0 = t0/100.0

        # mask valid
        mask = (~np.isnan(A_z)) & (~np.isnan(Y)) & (~np.isnan(t0))
        t_v = t0[mask]; A_v = A_z[mask]; Y_v = Y[mask]; RT_v = RT[mask]
        D_v = D_z[mask] if D_present else None
        print("after mask:", len(t_v), "trials")

        # descriptive save: comp_m hist
        plt.figure(figsize=(6,3))
        plt.hist(A[~np.isnan(A)], bins=12)
        plt.title(f"{scope_name} comp_m hist (n={np.sum(~np.isnan(A))})")
        save_fig(plt, f"{scope_name}_compm_hist.png")

        if len(t_v) < MIN_TRIALS_TO_FIT:
            print("not enough valid trials after mask")
            out[scope_name] = {"fitted": False, "reason":"not_enough_after_mask"}
            continue

        if not D_present:
            print("Fitting simplified model (amount only) with multi-start")
            params_best, loss_best = fit_simplified_multistart(t_v, A_v, Y_v, nstarts=MULTI_STARTS)
            print("best params:", params_best, "loss:", loss_best)
            v_hat, w1 = predict_v_simplified(params_best, t_v, A_v)
            p_hat = sigmoid(params_best[4]*v_hat)
            # binned plot
            plot_obs_pred_binned(t_v, Y_v, p_hat, scope_name, f"{scope_name}_obs_vs_pred.png")
            # weight curve
            tgrid = np.linspace(np.min(t_v), np.max(t_v), 200)
            wgrid = w_amt_t(tgrid, params_best[0], params_best[1], params_best[2])
            plot_weight_curve(tgrid, wgrid, "w_amt(t)", scope_name, f"{scope_name}_w_amt.png")
            # RT fit
            ndt, krt = fit_rt_given_v(v_hat, RT_v)
            if not np.isnan(ndt):
                rt_pred = ndt + krt/(np.abs(v_hat)+EPS_V)
                plt.figure(figsize=(5,5))
                plt.scatter(RT_v, rt_pred, alpha=0.7)
                mn, mx = np.nanmin(RT_v), np.nanmax(RT_v)
                plt.plot([mn,mx],[mn,mx],'k--')
                plt.title(f"{scope_name} RT obs vs pred")
                save_fig(plt, f"{scope_name}_rt_obs_pred.png")
            out[scope_name] = {"fitted": True, "model":"simplified", "params": {
                "alpha_amt":float(params_best[0]), "lam_amt":float(params_best[1]),
                "c_amt":float(params_best[2]), "bias":float(params_best[3]), "k_choice":float(params_best[4]),
                "loss":float(loss_best), "ndt":None if np.isnan(ndt) else float(ndt), "k_rt":None if np.isnan(krt) else float(krt)
            }}
        else:
            print("Fitting full model (amount + delay) with multi-start")
            params_best, loss_best = fit_full_multistart(t_v, A_v, D_v, Y_v, nstarts=MULTI_STARTS)
            print("best params:", params_best, "loss:", loss_best)
            v_hat, w1, w2 = predict_v_full(params_best, t_v, A_v, D_v)
            p_hat = sigmoid(params_best[8]*v_hat)
            plot_obs_pred_binned(t_v, Y_v, p_hat, scope_name, f"{scope_name}_obs_vs_pred.png")
            tgrid = np.linspace(np.min(t_v), np.max(t_v), 200)
            w1grid = w_amt_t(tgrid, params_best[0], params_best[1], params_best[2])
            w2grid = w_time_t(tgrid, params_best[3], params_best[4], params_best[5], params_best[6])
            plot_weight_curve(tgrid, w1grid, "w_amt(t)", scope_name, f"{scope_name}_w_amt.png")
            plot_weight_curve(tgrid, w2grid, "w_time(t)", scope_name, f"{scope_name}_w_time.png")
            # RT fit
            ndt, krt = fit_rt_given_v(v_hat, RT_v)
            if not np.isnan(ndt):
                rt_pred = ndt + krt/(np.abs(v_hat)+EPS_V)
                plt.figure(figsize=(5,5))
                plt.scatter(RT_v, rt_pred, alpha=0.7)
                mn, mx = np.nanmin(RT_v), np.nanmax(RT_v)
                plt.plot([mn,mx],[mn,mx],'k--')
                plt.title(f"{scope_name} RT obs vs pred")
                save_fig(plt, f"{scope_name}_rt_obs_pred.png")
            out[scope_name] = {"fitted": True, "model":"full", "params": params_best.tolist(), "loss": float(loss_best)}

    # save results
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results to", OUT_JSON)
    print("Done.")

if __name__=='__main__':
    main()
