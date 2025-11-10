# pip install yfinance pandas numpy matplotlib cvxpy osqp

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ======================================================
# CONFIGURACIÓN INICIAL Y DESCARGA DE DATOS
# ======================================================

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
START_DATE, END_DATE = "2019-01-01", "2025-01-01"
RISK_FREE = 0.04151  # Tasa libre de riesgo anual (bono 10Y USA)
np.random.seed(42)

# Descarga de precios y cálculo de retornos logarítmicos
prices = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"].dropna()
returns = np.log(prices / prices.shift(1)).dropna()

mu = returns.mean() * 252
cov = returns.cov() * 252
n_assets = len(TICKERS)

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================

def repair_weights(w: np.ndarray) -> np.ndarray:
    """Ajusta pesos a no negativos y que sumen 1."""
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() != 0 else np.ones_like(w) / len(w)

def portfolio_stats(weights, mu, cov, rf=0.0):
    """Calcula rendimiento, volatilidad y ratio de Sharpe."""
    w = np.array(weights)
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe

# ======================================================
# ALGORITMO GENÉTICO
# ======================================================

GA_PARAMS = {
    "pop_size": 100,
    "n_generations": 300,
    "cx_prob": 0.8,
    "mut_prob": 0.2,
    "mut_sigma": 0.05,
    "elite_size": 5,
    "tournament_k": 3,
}

def init_population(size, n_assets):
    """Inicializa una población de pesos válidos."""
    return np.array([repair_weights(np.random.rand(n_assets)) for _ in range(size)])

def fitness(weights, mu, cov, rf):
    """Función objetivo: maximizar Sharpe."""
    return portfolio_stats(weights, mu, cov, rf)[2]

def tournament_selection(pop, fitnesses, k):
    """Selecciona el mejor individuo de un torneo aleatorio."""
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(fitnesses[idx])]].copy()

def crossover(p1, p2):
    """Cruce lineal entre dos padres."""
    alpha = np.random.rand(len(p1))
    return repair_weights(alpha * p1 + (1 - alpha) * p2)

def mutate(ind, sigma=0.05, prob=0.2):
    """Aplica mutación gaussiana."""
    if np.random.rand() < prob:
        ind = repair_weights(ind + np.random.normal(0, sigma, ind.shape))
    return ind

def run_ga(mu, cov, rf, params=GA_PARAMS, generations=None):
    """Ejecuta el algoritmo genético y retorna el mejor individuo."""
    pop_size = params["pop_size"]
    n_generations = generations or params["n_generations"]
    pop = init_population(pop_size, len(mu))
    best_history = []

    for gen in range(n_generations):
        fits = np.array([fitness(ind, mu, cov, rf) for ind in pop])
        new_pop = []

        # Elitismo
        elites = pop[np.argsort(fits)[-params["elite_size"]:]]
        new_pop.extend(elites)

        # Reproducción
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits, params["tournament_k"])
            p2 = tournament_selection(pop, fits, params["tournament_k"])
            child = crossover(p1, p2) if np.random.rand() < params["cx_prob"] else p1.copy()
            child = mutate(child, params["mut_sigma"], params["mut_prob"])
            new_pop.append(child)

        pop = np.array(new_pop)
        best_fit = fits.max()
        best_history.append(best_fit)

        if (gen + 1) % 20 == 0:
            print(f"Gen {gen+1:03d} | Best Sharpe: {best_fit:.4f}")

    # Mejor individuo final
    fits = np.array([fitness(ind, mu, cov, rf) for ind in pop])
    best = pop[np.argmax(fits)]
    return best, best_history

# ======================================================
# ENTRENAMIENTO PRINCIPAL
# ======================================================

best_weights, best_history = run_ga(mu, cov, RISK_FREE)

ret, vol, sharpe = portfolio_stats(best_weights, mu, cov, RISK_FREE)
print("\n=== Mejor Solución GA ===")
for t, w in zip(TICKERS, best_weights):
    print(f"{t:6s}: {w:.4f}")
print(f"Retorno anual esperado: {ret:.2%}")
print(f"Volatilidad anual: {vol:.2%}")
print(f"Ratio de Sharpe: {sharpe:.4f}")

# Evolución del Sharpe
plt.figure(figsize=(8, 4))
plt.plot(best_history)
plt.title("Evolución del mejor Sharpe por generación")
plt.xlabel("Generación")
plt.ylabel("Sharpe (mejor individuo)")
plt.grid(True)
plt.show()

# Comparación con portafolio equiponderado
w_eq = np.ones(n_assets) / n_assets
ret_eq, vol_eq, sharpe_eq = portfolio_stats(w_eq, mu, cov, RISK_FREE)
print(f"\n=== Portafolio Equiponderado ===")
print(f"Retorno: {ret_eq:.2%} | Vol: {vol_eq:.2%} | Sharpe: {sharpe_eq:.4f}")

# ======================================================
# FRONTERAS EFICIENTES (Analítica, Muestreo, Exacta)
# ======================================================

# A) Frontera analítica (sin restricciones de short-selling)
ones = np.ones(n_assets)
inv_cov = np.linalg.pinv(cov.values)
A = ones @ inv_cov @ ones
B = ones @ inv_cov @ mu.values
C = mu.values @ inv_cov @ mu.values
D = A * C - B**2

target_returns = np.linspace(mu.min(), mu.max(), 200)
analytical_vols = np.sqrt(np.maximum((A * target_returns**2 - 2 * B * target_returns + C) / D, 0))

# B) Frontera aproximada (no short)
n_random = 20000
rand_weights = np.array([repair_weights(np.random.rand(n_assets)) for _ in range(n_random)])
rand_rets = rand_weights @ mu.values
rand_vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_weights, cov.values, rand_weights))
rand_sharpes = (rand_rets - RISK_FREE) / rand_vols

# C) Frontera exacta con CVXPY
try:
    import cvxpy as cp
    mu_grid = np.linspace(rand_rets.min(), rand_rets.max(), 80)
    exact_vols, exact_rets = [], []
    Sigma, mu_vec = cov.values, mu.values

    print("\nCalculando frontera exacta (cvxpy)...")
    for target in mu_grid:
        w = cp.Variable(n_assets)
        constraints = [mu_vec @ w == target, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        if w.value is not None:
            wv = repair_weights(np.array(w.value).flatten())
            r, v, _ = portfolio_stats(wv, mu, cov, RISK_FREE)
            exact_rets.append(r)
            exact_vols.append(v)
    exact_vols, exact_rets = np.array(exact_vols), np.array(exact_rets)
except Exception as e:
    print("Error calculando frontera exacta:", e)
    exact_vols, exact_rets = [], []

# Gráfico fronteras
plt.figure(figsize=(10, 7))
plt.scatter(rand_vols, rand_rets, c=rand_sharpes, cmap="viridis", s=8, alpha=0.4)
plt.plot(analytical_vols, target_returns, "r--", lw=2, label="Frontera analítica (sin restricciones)")
if len(exact_vols):
    plt.plot(exact_vols, exact_rets, "b", lw=2, label="Frontera exacta (cvxpy, no short)")
plt.scatter(vol, ret, marker="*", s=200, c="cyan", edgecolor="k", label="GA (mejor)")
plt.scatter(vol_eq, ret_eq, marker="o", s=100, c="magenta", edgecolor="k", label="Equiponderado")
plt.title("Fronteras eficientes: analítica, muestreo y exacta")
plt.xlabel("Volatilidad anualizada (σ)")
plt.ylabel("Retorno anualizado (μ)")
plt.legend()
plt.grid(True)
plt.show()

# ======================================================
# GRÁFICO paper GA Nube de portafolios
# ======================================================

import matplotlib as mpl
plt.style.use("default")
plt.grid(True, linestyle="--", alpha=0.6)

fig, ax = plt.subplots(figsize=(12, 8))

# Nube de portafolios aleatorios (color por Sharpe)
sc = ax.scatter(rand_vols, rand_rets,
                c=rand_sharpes,
                cmap="plasma",
                s=18,
                alpha=0.65,
                edgecolors="none")

# Punto del mejor portafolio GA
ax.scatter(vol, ret,
#           marker="*",
           s=50,
           c="#2a4dff",
#           edgecolor="k",
           linewidth=1.2,
           label="GA (Mejor)")

# Etiquetas y título
ax.set_title("Portafolios Aleatorios y Mejor Portafolio - Algoritmo Genético", fontsize=16, pad=12)
ax.set_xlabel("Volatilidad", fontsize=13)
ax.set_ylabel("Rendimiento Esperado", fontsize=13)

# Colorbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02)
cbar.set_label("Sharpe", rotation=270, labelpad=18)

# Leyenda y formato
ax.legend(loc="upper left", frameon=True, fontsize=11)
ax.tick_params(axis="both", which="major", labelsize=11)
ax.set_xlim(left=max(0, rand_vols.min() * 0.95), right=rand_vols.max() * 1.05)
ax.set_ylim(bottom=rand_rets.min() * 0.95, top=rand_rets.max() * 1.08)

plt.show()

# ======================================================
# VALIDACIÓN WALK-FORWARD
# ======================================================

print("\n=== Validación Walk-Forward ===")
window_size = int(len(returns) * 0.6)
step_size = int(len(returns) * 0.1)
cv_results, start, fold = [], 0, 1

while start + window_size + step_size <= len(returns):
    train = returns.iloc[start:start + window_size]
    test = returns.iloc[start + window_size:start + window_size + step_size]

    mu_train, cov_train = train.mean() * 252, train.cov() * 252
    mu_test, cov_test = test.mean() * 252, test.cov() * 252

    best_w, _ = run_ga(mu_train, cov_train, RISK_FREE, generations=100)
    ret_t, vol_t, sharpe_t = portfolio_stats(best_w, mu_test, cov_test, RISK_FREE)
    cv_results.append([ret_t, vol_t, sharpe_t])

    print(f"Fold {fold:02d}: {test.index[0].date()} - {test.index[-1].date()} | Sharpe={sharpe_t:.4f}")
    start += step_size
    fold += 1

cv_results = np.array(cv_results)
mean_ret, mean_vol, mean_sharpe = cv_results.mean(axis=0)
std_sharpe = cv_results[:, 2].std()

print("\nResultados promedio (Walk-Forward):")
print(f"Retorno medio: {mean_ret:.2%}")
print(f"Volatilidad media: {mean_vol:.2%}")
print(f"Sharpe medio: {mean_sharpe:.4f} ± {std_sharpe:.4f}")