import numpy as np
from scipy.stats.mstats import mquantiles
from sklearn.metrics import pairwise_distances, roc_auc_score, r2_score, pairwise_distances_argmin_min
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import DescrStatsW
from statistics import mode
from tqdm import tqdm
from sklearn.base import clone
import random
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture


def projection_simplexe(y):
    """
    Projection euclidienne d'un vecteur y sur le simplexe Δ = {x >= 0, sum(x) = 1}
    Implémentation directe de l'algorithme de Duchi et al. (2008).
    """
    # 1) Trier y en ordre décroissant
    u = np.sort(y)[::-1]
    # 2) Trouver rho
    cssv = np.cumsum(u)
    m = np.arange(1, len(y)+1)
    cond = u - (cssv - 1) / m > 0
    if np.any(cond):
      rho = np.nonzero(cond)[0][-1]  # [0] donne les indices ou la cond est true , [-1] dernier indice qui vérifie la condition=max indices m
      # 3) Calculer theta
      theta = (cssv[rho] - 1) / (rho + 1) #+1 car indice python commence a 0, pas a 1
    else:
      # si aucune valeur ne satisfait cond, prendre theta=0
      theta = 0
    
    # 4) Sortie
    x = np.maximum(y - theta, 0)
    return x

def generer_F_gaussiennes_residus(r, K=3, sigma=0.1):
    """
    Génère une matrice F (n,K) où chaque colonne est une gaussienne
    évaluée sur les résidus r, avec des moyennes espacées et un sigma fixé.
    
    - r : vecteur des résidus (taille n)
    - K : nombre de gaussiennes
    - sigma : écart-type des gaussiennes
    """
    r = np.array(r)
    n = len(r)

    # Choisir K centres régulièrement espacés entre min(r) et max(r)
    mu = np.linspace(r.min(), r.max(), K)

    # Construire la matrice F
    F = np.zeros((n, K))
    for k in range(K):
        F[:, k] = norm.pdf(r, loc=mu[k], scale=sigma)

    return F, mu

def generer_F_gaussiennes_auto(r, K=None, sigma=None):
    r = np.array(r)
    n = len(r)

    # Si K non fourni → choisir automatiquement (nb unique valeurs borné par 5)
    if K is None:
        K = min(len(np.unique(r)), 5)  # max 5 gaussiennes pour éviter la sursegmentation

    # Centres espacés régulièrement
    mu = np.linspace(r.min(), r.max(), K)

    if sigma is None:
        if K > 1:
            sigma = (mu[1] - mu[0]) / 2.0
        else:
            sigma = 1.0  # fallback

    # Construire la matrice F
    F = np.zeros((n, K))
    for k in range(K):
        F[:, k] = norm.pdf(r, loc=mu[k], scale=sigma)

    return F, mu

def generer_F_gaussiennes_residus_gmm(r, K=3):
    r = np.array(r).reshape(-1, 1)
    n = len(r)

    gmm = GaussianMixture(n_components=min(K, len(np.unique(r))), random_state=42).fit(r)
    mu = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())

    F = np.zeros((n, len(mu)))
    for k in range(len(mu)):
        F[:, k] = norm.pdf(r[:, 0], loc=mu[k], scale=stds[k])

    return F, mu


def algo_sousgradient_projete_residus(r, X, K=2,lambda_pen=10, eta0=0.01, max_iter=20, tol=0.005):
    """
    Algorithme de sous-gradient projeté :
    - F construit à partir des résidus r
    - W construit à partir des covariables X (multidimensionnel)
    Entrées :
      r : vecteur des résidus (n,)
      X : matrice des covariables standardisées (n,d)
      K : nb de gaussiennes
      sigma : écart-type des gaussiennes
      lambda_pen : paramètre λ
      eta0 : pas initial
      T : nombre max d’itérations
      tol : tolérance d’arrêt
    """
    n, d = X.shape
    # Étape 1 : Construire F à partir des résidus
    # F, mu = generer_F_gaussiennes_residus(r, K=K,sigma=sigma)
    F,mu=generer_F_gaussiennes_auto(r,K=K)
    # Étape 2 : Initialisation
    pi = np.full((n, K), 1/K)
    # Étape 3 : Matrice des distances et poids basée sur X
    dist = cdist(X, X, metric='euclidean')  # distances entre xi et xj
    np.fill_diagonal(dist, np.inf)
    W = 1 / dist

    history_delta = []
    # Étape 4 : Boucle
    for t in range(max_iter):
        eta_t = eta0 / np.sqrt(t + 1)
        pi_old = pi.copy()
        for i in range(n):
            # (a) Gradient de la partie lisse
            Z_i = np.dot(pi[i], F[i])
            g_lisse = -F[i] / Z_i
            # (b) Sous-gradient de la pénalité
            diff = pi[i] - pi
            g_pen = (W[i][:, None] * np.sign(diff)).sum(axis=0) / lambda_pen
            # (c) Gradient total
            g_total = g_lisse + g_pen
            # (d-e) Descente + projection
            y = pi[i] - eta_t * g_total
            pi[i] = projection_simplexe(y)
        # (3) Critère d'arrêt
        delta = np.max(np.sum(np.abs(pi - pi_old), axis=1))# max_i ||π^{i,t+1}-π^{i,t}||_1
        history_delta.append(delta)

        if delta <= tol:
            print(f"Convergence atteinte en {t+1} itérations (variation={delta:.2e}).")
            break
    
    return pi, mu, F, history_delta


def approx_pi_test(prob, X_cal, X_test, max_iter=20, eta0=0.01, tol=0.005):
    """
    prob: array (n,K) vecteurs pi connus
    X_cal : array (n,d) points calibration
    X_test : array (m,d) points test
    Retourne : array (m,K) des pi approximés pour chaque point test
    """
    n, K = prob.shape
    m = X_test.shape[0]
    pi_all = np.zeros((m,K))
    # Étape 1 : Matrice distances n x m
    dist = cdist(X_cal, X_test, metric='euclidean')  # n x m
    epsilon = 1e-8
    W = 1 / (dist + epsilon)
    # Étape 2 : boucle sur chaque point test
    for j in range(m):
        w = W[:, j]              # poids pour le point test j
        p = np.ones(K)/K          # initialisation
        prev_obj = np.sum(w[:,None] * np.abs(p - prob))
        # descente projetée
        for t in range(max_iter):
            g = np.sum(w[:,None] * np.sign(p - prob), axis=0)
            eta = eta0 / np.sqrt(t+1)
            p = projection_simplexe(p - eta*g)
            obj = np.sum(w[:,None] * np.abs(p - prob))
            if abs(prev_obj - obj) < tol:
                break
            prev_obj = obj

        pi_all[j] = p

    return pi_all

def calibrate_pcp_split(X_val,R_val,X_test,R_test,n_cluster,m,alpha,return_pi=False, finite=False, max_iter=10, tol=0.005,lambda_pen=10):
    #On veut estimer les poids par une descente de gradient projeté
    # Standardize the features.
    prob, mu, _,_= algo_sousgradient_projete_residus(R_val,X_val,K=n_cluster,max_iter=max_iter, tol=tol,lambda_pen=lambda_pen)
    pi_test_all=approx_pi_test(prob,X_val,X_test)
    intervals = []
    for j,x in enumerate(X_test):
        idx = random.choices(population=range(n_cluster), weights=pi_test_all[j], k=m)
        # Calcul des log-vraisemblances cumulées pour tous les points + X_{n+1}
        # On concatène les lignes de prob + pi_test_all[j] pour inclure X_{n+1}
        prob_extended = np.vstack([prob, pi_test_all[j]])  # shape (n_val+1, n_cluster)
        s = np.sum(np.log(prob_clip(prob_extended)[:, idx]), axis=1)  # shape (n_val+1,)
        # Stabilisation numérique + exponentielle
        max_s = np.max(s)
        weight = np.exp(s - max_s)
        weight /= np.sum(weight)
        bigM = np.inf
        R_aug = np.append(R_val, bigM)
        wq = DescrStatsW(data=R_aug, weights=weight)
        r = wq.quantile(probs=1 - alpha, return_pandas=False)[0]
        intervals.append(r)
    if return_pi:
        return np.array(intervals), pi_test_all
    else:
        return np.array(intervals)

#C_n(X_{n+1}=mu(X_test)+-quantile_(1_alpha)(sum(wi*delta_Ri)+wn+1*delta_inf))

def determine_precision(n, prob,n_c):
    """Determine the sample size m for generating the PCP interval."""
    m_max, m_min = 501, 4
    m_ = (m_max + m_min) / 2

    seeds = np.random.randint(1, 100000001, size=min(n, 1000))
    while abs(m_max - m_min) > 2:
        w_w_0 = np.zeros((min(n, 1000), min(n, 1000)))

        for j in range(min(n, 1000)):
            random.seed(int(seeds[j]))
            idx = random.choices(population=range(n_c), weights=prob[j, :], k=int(m_))
            w_w_0[j, :] = np.sum(np.log(prob_clip(prob)[:min(n, 1000), idx]), axis=1)
            w_j = w_w_0[j, :]
            max_w = np.max(w_j)
            w_j = np.exp(w_j - max_w)
            w_j /= np.sum(w_j)
            w_w_0[j, :] = w_j

        n_hat = np.mean(1 / np.sum(w_w_0 ** 2, axis=1)) * min(max(n / 1000, 1), 1)
        n_hat_2 = np.mean(np.diagonal(w_w_0))

        if n_hat <= 100 and n_hat_2 >= 1 / 30:
            m_max = m_
        else:
            m_min = m_

        m_ = (m_max + m_min) / 2


def prob_clip(prob, epsilon=1e-8):
    return np.maximum(prob, epsilon)


def prob_smoother(prob, epsilon=0.025):
    return (prob + epsilon) / (1 + np.shape(prob)[1] * epsilon)



def pcp_full_prediction_intervals(X_test, predictions,X_val, R_val,R_test,grids_per_test,m,n_cluster, alpha=0.1,max_iter=50, tol=1e-3,lambda_pen=100):
    """
    Construit les intervalles de prédiction PCP pour tous les points test.
    intervals : list of tuples
        intervalles [borne_inf, borne_sup] pour chaque point test
    """
    intervals = []

    for i, x_test in enumerate(X_test):
        grid = grids_per_test[i]
        pred_test = predictions[i]
        accepted_y = []
        coverage = []
        for y in grid:
            # 1) résidu pour ce y
            R_y = abs(pred_test - y)

            # 2) concaténer données
            R_aug = np.append(R_val, R_y)
            X_aug = np.vstack([X_val, x_test])

            # 3) estimer poids
            prob, mu, _, _ = algo_sousgradient_projete_residus(R_aug, X_aug, K=n_cluster, 
                                       max_iter=max_iter, tol=tol,lambda_pen=lambda_pen)
            
            # prob_train = pi_train
            pi_train = prob[:-1]  # validation
            pi_test = prob[-1]    # point tes
            # 4) approx Monte Carlo pour le point test
            idx = random.choices(population=range(n_cluster), weights=pi_test, k=m)
            prob_extended = np.vstack([pi_train, pi_test])
            s = np.sum(np.log(prob_clip(prob_extended)[:, idx]), axis=1)
            max_s = np.max(s)
            weight = np.exp(s - max_s)
            weight /= np.sum(weight)
            wq = DescrStatsW(data=np.append(R_val, R_y), weights=weight)
            r = wq.quantile(probs=1 - alpha, return_pandas=False)[0]
            # Stop if the weighted quantile is within the grid
            if r >= R_y:
                coverage.append(R_test[i] <= r)
                intervals.append(r)
                break
            # 4) calcul p-value
        #     p_val = np.sum(weight * (R_aug >= R_y))

        #     if p_val > alpha:
        #         accepted_y.append(y)

        # if len(accepted_y) == 0:
        #     # pas de y accepté -> intervalle vide
        #     intervals.append((np.nan, np.nan))
        # else:
        #     intervals.append((min(accepted_y), max(accepted_y)))

    return intervals,coverage



# def compute_weights(pi_train, pi_test, m=1.0, eps=1e-12):
#     """
#     Calcule les poids PCP (w_i) pour un point test.
#     pi_train : array (n, K)  - distributions du mélange pour les points d'entraînement
#     pi_test  : array (K,)    - distribution du mélange pour le point test
#     m        : float         - paramètre m (>=0)
#     eps      : float         - évite les divisions/log de 0
#     """
#     pi_train = np.clip(pi_train, eps, 1.0)
#     pi_test = np.clip(pi_test, eps, 1.0)
    
#     # normaliser
#     pi_train = pi_train / pi_train.sum(axis=1, keepdims=True)
#     pi_test = pi_test / pi_test.sum()
    
#     # KL(pi_test || pi_i)
#     kl = np.sum(pi_test * (np.log(pi_test) - np.log(pi_train)), axis=1)
    
#     # poids non normalisés
#     unnorm = np.exp(-m * kl)
#     return unnorm / unnorm.sum()




