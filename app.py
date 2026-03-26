import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm, multivariate_normal
from dataclasses import dataclass
from time import perf_counter

# ──────────────────────────────────────────
# Core Building Blocks
# ──────────────────────────────────────────

def black_call(F, K, vol_sqrtT):
    F = float(F); K = float(K)
    if vol_sqrtT <= 1e-12:
        return max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * vol_sqrtT**2) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def margrabe(F1, F2, sigma1, sigma2, rho, T):
    v = (sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2) * T
    vs = np.sqrt(max(v, 0.0))
    if vs <= 1e-14:
        return max(F2 - F1, 0.0)
    d1 = (np.log(F2/F1) + 0.5*v) / vs
    d2 = d1 - vs
    return F2 * norm.cdf(d1) - F1 * norm.cdf(d2)


def kirk(F1, F2, sigma1, sigma2, rho, T, K=0.0):
    if abs(K) < 1e-15:
        return margrabe(F1, F2, sigma1, sigma2, rho, T)
    beta = F2 / (F2 + K)
    v = (sigma1**2 - 2*beta*rho*sigma1*sigma2 + (beta**2)*sigma2**2) * T
    vs = np.sqrt(max(v, 0.0))
    d1 = (np.log(F1/(F2+K)) + 0.5*v) / vs
    d2 = d1 - vs
    return F1 * norm.cdf(d1) - (F2 + K) * norm.cdf(d2)


def lzd_conditional(F1, F2, sigma1, sigma2, rho, T, K=0.0, n=32):
    z, w = hermgauss(n)
    mu1 = np.log(F1) - 0.5 * (sigma1**2) * T
    s1 = sigma1 * np.sqrt(T)
    v2_cond = (1 - rho**2) * sigma2**2 * T
    total = 0.0
    for zi, wi in zip(z, w):
        lnS1 = mu1 + np.sqrt(2.0) * s1 * zi
        S1 = np.exp(lnS1)
        mu2_given1 = (np.log(F2) - 0.5*sigma2**2*T) + (rho * sigma2 / sigma1) * (lnS1 - mu1)
        F2_cond = np.exp(mu2_given1 + 0.5 * v2_cond)
        K_eff = S1 + K
        if K_eff <= 1e-15:
            price_cond = F2_cond
        else:
            price_cond = black_call(F2_cond, K_eff, np.sqrt(v2_cond))
        total += wi * price_cond
    return float(total / np.sqrt(np.pi))


def outrank_prob(Fi, Fj, si, sj, rho_ij, T):
    mu = (np.log(Fi) - 0.5*si**2*T) - (np.log(Fj) - 0.5*sj**2*T)
    var = (si**2 + sj**2 - 2*rho_ij*si*sj) * T
    s = np.sqrt(max(var, 1e-18))
    return float(norm.cdf(mu / s))


@dataclass
class Market:
    F: np.ndarray
    sigma: np.ndarray
    rho: np.ndarray
    T: float = 1.0
    K: float = 0.0
    r: float = 0.0

    def check(self):
        m = len(self.F)
        assert self.sigma.shape == (m,), 'sigma shape mismatch'
        assert self.rho.shape == (m, m), 'rho shape mismatch'
        ev = np.linalg.eigvalsh(self.rho)
        if ev.min() < -1e-6:
            raise ValueError('Correlation matrix not PSD')


def joint_prob_S4_beats_S2_S3(F, sigma, R, T):
    sig = np.array(sigma)
    var = (sig**2) * T
    mu = np.log(F) - 0.5 * var
    C = (sig[:, None] * sig[None, :] * R) * T
    i2, i3, i4 = 1, 2, 3
    m1 = mu[i4] - mu[i2]
    m2 = mu[i4] - mu[i3]
    v4 = var[i4]; v2 = var[i2]; v3 = var[i3]
    c42 = C[i4, i2]; c43 = C[i4, i3]; c23 = C[i2, i3]
    s11 = v4 + v2 - 2.0*c42
    s22 = v4 + v3 - 2.0*c43
    s12 = v4 - c43 - c42 + c23
    Sigma = np.array([[s11, s12], [s12, s22]], dtype=float)
    muY = np.array([m1, m2], dtype=float)
    p_y1_le0 = norm.cdf(-muY[0] / np.sqrt(max(Sigma[0, 0], 0.0)))
    p_y2_le0 = norm.cdf(-muY[1] / np.sqrt(max(Sigma[1, 1], 0.0)))
    try:
        c00 = multivariate_normal(mean=muY, cov=Sigma).cdf([0.0, 0.0])
        return float(1.0 - p_y1_le0 - p_y2_le0 + c00)
    except Exception:
        rng = np.random.default_rng(12345)
        Z = rng.multivariate_normal(mean=muY, cov=Sigma, size=400_000)
        return float(np.mean((Z[:, 0] > 0) & (Z[:, 1] > 0)))


# ──────────────────────────────────────────
# Pricing Models
# ──────────────────────────────────────────

def presentation_price(mkt: Market):
    mkt.check()
    F, s, R, T, K = mkt.F, mkt.sigma, mkt.rho, mkt.T, mkt.K
    base = 0
    alts = np.argsort(-F[1:]) + 1

    def spread(i):
        return kirk(F[base], F[i], s[base], s[i], R[base, i], T, K)

    price = 0.0; terms = []
    if len(alts) >= 1:
        p2 = spread(alts[0]); price += p2; terms.append((int(alts[0]+1), p2, 1.0))
    if len(alts) >= 2:
        i3 = alts[1]; p3 = spread(i3)
        P32 = outrank_prob(F[i3], F[alts[0]], s[i3], s[alts[0]], R[i3, alts[0]], T)
        price += p3 * P32; terms.append((int(i3+1), p3, P32))
    if len(alts) >= 3:
        i4 = alts[2]; p4 = spread(i4)
        P42 = outrank_prob(F[i4], F[alts[0]], s[i4], s[alts[0]], R[i4, alts[0]], T)
        P43 = outrank_prob(F[i4], F[alts[1]], s[i4], s[alts[1]], R[i4, alts[1]], T)
        price += p4 * P42 * P43; terms.append((int(i4+1), p4, P42*P43))
    return price, terms


def lzd_ext_price(mkt: Market, n_gh: int = 32):
    mkt.check()
    F, s, R, T, K = mkt.F, mkt.sigma, mkt.rho, mkt.T, mkt.K
    base = 0
    alts = np.argsort(-F[1:]) + 1

    def spread_lzd(i):
        return lzd_conditional(F[base], F[i], s[base], s[i], R[base, i], T, K, n=n_gh)

    price = 0.0; terms = []
    if len(alts) >= 1:
        p2 = spread_lzd(alts[0]); price += p2; terms.append((int(alts[0]+1), p2, 1.0))
    if len(alts) >= 2:
        i3 = alts[1]; p3 = spread_lzd(i3)
        P32 = outrank_prob(F[i3], F[alts[0]], s[i3], s[alts[0]], R[i3, alts[0]], T)
        price += p3 * P32; terms.append((int(i3+1), p3, P32))
    if len(alts) >= 3:
        i4 = alts[2]; p4 = spread_lzd(i4)
        P42 = outrank_prob(F[i4], F[alts[0]], s[i4], s[alts[0]], R[i4, alts[0]], T)
        P43 = outrank_prob(F[i4], F[alts[1]], s[i4], s[alts[1]], R[i4, alts[1]], T)
        price += p4 * P42 * P43; terms.append((int(i4+1), p4, P42*P43))
    return price, terms


def presentation_mvn_price(mkt: Market):
    mkt.check()
    F, s, R, T, K = mkt.F, mkt.sigma, mkt.rho, mkt.T, mkt.K
    base = 0
    alts = np.argsort(-F[1:]) + 1

    def spread(i):
        return kirk(F[base], F[i], s[base], s[i], R[base, i], T, K)

    price = 0.0; terms = []
    if len(alts) >= 1:
        p2 = spread(alts[0]); price += p2; terms.append((int(alts[0]+1), p2, 1.0))
    if len(alts) >= 2:
        i3 = alts[1]; p3 = spread(i3)
        P32 = outrank_prob(F[i3], F[alts[0]], s[i3], s[alts[0]], R[i3, alts[0]], T)
        price += p3 * P32; terms.append((int(i3+1), p3, P32))
    if len(alts) >= 3:
        i2 = alts[0]; i4 = alts[2]
        p4 = spread(i4)
        F4 = np.array([F[base], F[i2], F[alts[1]], F[i4]], dtype=float)
        s4 = np.array([s[base], s[i2], s[alts[1]], s[i4]], dtype=float)
        R4 = R[[base, i2, alts[1], i4]][:, [base, i2, alts[1], i4]].astype(float)
        P4_joint = joint_prob_S4_beats_S2_S3(F4, s4, R4, T)
        price += p4 * P4_joint; terms.append((int(i4+1), p4, P4_joint))
    return price, terms


def mc_price(mkt: Market, n_sims: int = 100_000, antithetic: bool = True, seed: int = 7):
    mkt.check()
    F, s, R, T, K = mkt.F, mkt.sigma, mkt.rho, mkt.T, mkt.K
    rng = np.random.default_rng(seed)
    m = len(F)
    L = np.linalg.cholesky(R + 1e-12*np.eye(m))
    n = (n_sims + 1)//2 if antithetic else n_sims
    Z = rng.standard_normal((n, m))
    if antithetic:
        Z = np.vstack([Z, -Z])
    Z = Z[:n_sims]
    shocks = Z @ L.T
    lnS = np.log(F) + (-0.5 * s**2 * T) + shocks * (s * np.sqrt(T))
    S = np.exp(lnS)
    payoff = np.maximum(S[:, 1:] - S[:, 0:1] - K, 0.0)
    return payoff.max(axis=1).mean()


# ──────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────

st.set_page_config(page_title="Rainbow Options – MVN Comparison", layout="wide")
st.title("Rainbow Options: MVN-aware Pricing Comparison")
st.markdown(
    "Compare **Presentation**, **LZD-ext**, **Presentation-MVN**, and **Monte Carlo** "
    "pricing for best-of rainbow options (3 or 4 assets)."
)

# ── Sidebar: parameters ──────────────────
st.sidebar.header("Market Parameters")

n_assets = st.sidebar.radio("Number of assets", [3, 4], index=1)

st.sidebar.subheader("Forwards")
if n_assets == 4:
    F_defaults = [50.0, 52.0, 51.0, 49.5]
else:
    F_defaults = [50.0, 52.0, 51.0]

F_vals = [
    st.sidebar.number_input(f"F{i+1}", value=F_defaults[i], step=0.5, format="%.2f")
    for i in range(n_assets)
]

st.sidebar.subheader("Volatilities")
if n_assets == 4:
    sig_defaults = [0.35, 0.30, 0.30, 0.32]
else:
    sig_defaults = [0.35, 0.30, 0.30]

sig_vals = [
    st.sidebar.number_input(f"σ{i+1}", value=sig_defaults[i], min_value=0.01, max_value=2.0, step=0.01, format="%.2f")
    for i in range(n_assets)
]

st.sidebar.subheader("Option Parameters")
T = st.sidebar.number_input("Maturity T (years)", value=0.5, min_value=0.01, max_value=5.0, step=0.1, format="%.2f")
K = st.sidebar.number_input("Strike K (spread strike)", value=0.0, step=0.5, format="%.2f")

st.sidebar.subheader("Monte Carlo")
n_sims = st.sidebar.select_slider("MC simulations", options=[50_000, 100_000, 200_000, 300_000, 500_000], value=200_000)
mc_seed = st.sidebar.number_input("MC seed", value=42, min_value=0, step=1)

# ── Correlation matrix ───────────────────
st.subheader("Correlation Matrix")
st.markdown("Edit pairwise correlations (matrix is symmetrised automatically).")

if n_assets == 4:
    rho_defaults = np.array([
        [1.0,  0.6,  0.5,  0.55],
        [0.6,  1.0,  0.55, 0.5 ],
        [0.5,  0.55, 1.0,  0.6 ],
        [0.55, 0.5,  0.6,  1.0 ],
    ])
else:
    rho_defaults = np.array([
        [1.0, 0.6, 0.6],
        [0.6, 1.0, 0.6],
        [0.6, 0.6, 1.0],
    ])

cols = st.columns(n_assets)
rho_input = np.eye(n_assets)
for i in range(n_assets):
    for j in range(n_assets):
        if i == j:
            rho_input[i, j] = 1.0
        elif j > i:
            val = cols[j].number_input(
                f"ρ({i+1},{j+1})",
                value=float(rho_defaults[i, j]),
                min_value=-0.99, max_value=0.99, step=0.05, format="%.2f",
                key=f"rho_{i}_{j}",
            )
            rho_input[i, j] = val
            rho_input[j, i] = val

# PSD check
ev = np.linalg.eigvalsh(rho_input)
if ev.min() < -1e-6:
    st.error(f"Correlation matrix is **not positive semi-definite** (min eigenvalue {ev.min():.4f}). Adjust correlations.")
    st.stop()

# ── Run pricing ──────────────────────────
st.divider()
run = st.button("Run Pricing", type="primary")

if run:
    F_arr = np.array(F_vals, dtype=float)
    s_arr = np.array(sig_vals, dtype=float)
    mkt = Market(F=F_arr, sigma=s_arr, rho=rho_input, T=T, K=K)

    with st.spinner("Computing prices…"):
        t0 = perf_counter(); v_pres, terms_pres = presentation_price(mkt);      t1 = perf_counter()
        t2 = perf_counter(); v_lzd,  terms_lzd  = lzd_ext_price(mkt, n_gh=32); t3 = perf_counter()
        t4 = perf_counter(); v_mc                = mc_price(mkt, n_sims=n_sims, antithetic=True, seed=mc_seed); t5 = perf_counter()

        if n_assets == 4:
            t6 = perf_counter(); v_pmvn, terms_pmvn = presentation_mvn_price(mkt); t7 = perf_counter()
        else:
            v_pmvn = None; t6 = t7 = None

    # ── Results table ────────────────────
    st.subheader("Results")
    row = {
        "Presentation": v_pres,
        "LZD-ext": v_lzd,
        "Pres-MVN": v_pmvn if n_assets == 4 else float("nan"),
        "Monte Carlo": v_mc,
        "t_pres (ms)": (t1-t0)*1e3,
        "t_lzd (ms)": (t3-t2)*1e3,
        "t_pmvn (ms)": (t7-t6)*1e3 if n_assets == 4 else float("nan"),
        "t_mc (ms)": (t5-t4)*1e3,
    }
    df_res = pd.DataFrame([row])
    df_res["Pres vs MC (%)"]  = 100*(df_res["Presentation"] / df_res["Monte Carlo"] - 1)
    df_res["LZD vs MC (%)"]   = 100*(df_res["LZD-ext"]      / df_res["Monte Carlo"] - 1)
    if n_assets == 4:
        df_res["PMVN vs MC (%)"] = 100*(df_res["Pres-MVN"]  / df_res["Monte Carlo"] - 1)

    st.dataframe(df_res.round(5), use_container_width=True)

    # ── Bar charts ───────────────────────
    st.subheader("Value & Runtime Comparison")
    labels = ["Presentation", "LZD-ext", "Monte Carlo"] if n_assets == 3 else ["Presentation", "LZD-ext", "Pres-MVN", "Monte Carlo"]
    values = [v_pres, v_lzd, v_mc] if n_assets == 3 else [v_pres, v_lzd, v_pmvn, v_mc]
    times  = [(t1-t0)*1e3, (t3-t2)*1e3, (t5-t4)*1e3] if n_assets == 3 else [(t1-t0)*1e3, (t3-t2)*1e3, (t7-t6)*1e3, (t5-t4)*1e3]
    colors = ["#2a9d8f", "#264653", "#8ab17d", "#e76f51"][:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, values, color=colors)
    axes[0].set_title("Option Values"); axes[0].set_ylabel("Value"); axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, times, color=colors)
    axes[1].set_title("Runtime (ms)"); axes[1].set_ylabel("Milliseconds"); axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Term breakdown ───────────────────
    if n_assets >= 3:
        st.subheader("Term Breakdown")
        def terms_df(terms, label):
            return pd.DataFrame(
                [(f"Asset {t[0]}", f"{t[1]:.5f}", f"{t[2]:.5f}", f"{t[1]*t[2]:.5f}") for t in terms],
                columns=["Asset", "Spread price", "Rank weight", "Contribution"],
            ).assign(Method=label)

        dfs = [terms_df(terms_pres, "Presentation"), terms_df(terms_lzd, "LZD-ext")]
        if n_assets == 4:
            dfs.append(terms_df(terms_pmvn, "Pres-MVN"))
        st.dataframe(pd.concat(dfs, ignore_index=True), use_container_width=True)

    # ── Correlation stress (4-asset only) ─
    if n_assets == 4:
        st.subheader("Stress: Absolute Error vs Uniform Correlation")
        st.markdown("Sweeps a uniform correlation *c* ∈ [0, 0.9] while keeping all other parameters fixed.")

        n_stress_sims = st.select_slider(
            "Stress MC simulations", options=[50_000, 100_000, 200_000], value=100_000, key="stress_sims"
        )

        with st.spinner("Running correlation sweep…"):
            corr_levels = np.linspace(0.0, 0.9, 10)
            err_pres_arr, err_lzd_arr, err_pmvn_arr = [], [], []

            for c in corr_levels:
                R_c = np.full((4, 4), c); np.fill_diagonal(R_c, 1.0)
                mk = Market(F=F_arr, sigma=s_arr, rho=R_c, T=T, K=K)
                vp, _ = presentation_price(mk)
                vl, _ = lzd_ext_price(mk)
                vm, _ = presentation_mvn_price(mk)
                vM    = mc_price(mk, n_sims=n_stress_sims, antithetic=True, seed=777)
                err_pres_arr.append(abs(vp - vM))
                err_lzd_arr.append(abs(vl - vM))
                err_pmvn_arr.append(abs(vm - vM))

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(corr_levels, err_pres_arr, label="Presentation abs error", color="#2a9d8f")
        ax2.plot(corr_levels, err_lzd_arr,  label="LZD-ext abs error",      color="#264653")
        ax2.plot(corr_levels, err_pmvn_arr, label="Pres-MVN abs error",     color="#e76f51", linewidth=2)
        ax2.set_xlabel("Uniform correlation c"); ax2.set_ylabel("Absolute error vs MC")
        ax2.set_title("4-asset: Error vs correlation (uniform PSD)")
        ax2.grid(True, alpha=0.3); ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
else:
    st.info("Configure parameters in the sidebar and press **Run Pricing**.")
