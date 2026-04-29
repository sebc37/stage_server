"""
run_goy.py — Reproduit exactement la simulation C et compare avec le fichier de reference.

Parametres (dt=1e-5, fs=100, T=1000) :
  count_init = int(time/dt) = int(1000/1e-5) = 99999999
  N_fs       = int(1/dt/fs) = int(1/1e-5/100) = 999
  ligne 0    = etat apres 100 pas  (premier count%999==0)
  ligne k    = etat apres 100 + k*999 pas
"""

import numpy as np
import matplotlib.pyplot as plt
from goy import GoyModel

# ── parametres (identiques a parameters.h) ────────────────────────────────────
TIME      = 1000.
DT        = 1e-5
FS        = 100.
FORCE     = 0.005
N_FORCE   = 4
FORCE_RND = 0

REF_FILE  = "/home/s26calme/Documents/code_stage/GOY-main/data_test_precis.dat"

# ── replication exacte des calculs entiers du code C ─────────────────────────
count_init   = int(TIME / DT)           # 99999999
N_fs         = int(1.0 / DT / FS)      # 999
n_steps_first = count_init % N_fs      # 99  (reste avant premier multiple de 999)
if n_steps_first == 0:
    n_steps_first = N_fs
N_rows = count_init // N_fs             # 100100

print(f"count_init    = {count_init}")
print(f"N_fs          = {N_fs}")
print(f"Pas ligne 0   = {n_steps_first}")
print(f"Pas suivants  = {N_fs}")
print(f"Nombre lignes = {N_rows}")

# ── initialisation ────────────────────────────────────────────────────────────
model = GoyModel(dt=DT, force=FORCE, N_force=N_FORCE, force_rnd=FORCE_RND)
N = model.N  # 22

Xpp, Ypp, Xp, Yp = model.init_fields()
cur_Xpp, cur_Ypp = Xpp.copy(), Ypp.copy()
cur_Xp,  cur_Yp  = Xp.copy(),  Yp.copy()

state = np.zeros((N_rows, 2 * N))

# ── ligne 0 : n_steps_first pas ───────────────────────────────────────────────
(cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
    cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=n_steps_first)
state[0, 0::2] = cur_Xp
state[0, 1::2] = cur_Yp

# ── boucle principale : N_fs pas par ligne ────────────────────────────────────
for i in range(1, N_rows):
    if i % 10000 == 0:
        print(f"  ligne {i}/{N_rows} ...")
    (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
        cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=N_fs)
    state[i, 0::2] = cur_Xp
    state[i, 1::2] = cur_Yp

# ── comparaison avec le fichier de reference ──────────────────────────────────
try:
    ref = np.loadtxt(REF_FILE,dtype=np.float64)
    
    diff_X = state[:, 0::2] - ref[:, 0::2]
    diff_Y = state[:, 1::2] - ref[:, 1::2]
    for i in range(4):
        # plt.figure()
        # plt.plot(diff_X[:,i])
        # plt.plot(diff_Y[:,i])
        plt.figure()
        plt.plot(state[:,i])
        plt.plot(ref[:,i])
        plt.figure()
        plt.plot(state[:,i])
        plt.plot(ref[:,i])
    plt.show()
    print(f"\n=== Comparaison sur tout lignes ===")
    print(f"Max |ΔX| = {np.max(np.abs(diff_X)):.3e}")
    print(f"Max |ΔY| = {np.max(np.abs(diff_Y)):.3e}")
    print(f"RMSE X   = {np.sqrt(np.mean(diff_X**2)):.3e}")
    print(f"RMSE Y   = {np.sqrt(np.mean(diff_Y**2)):.3e}")

    # divergence ligne par ligne sur shell 0
    print(f"\nEvolution de |ΔX[0]| par ligne (10 premieres) :")
    # for i in range(min(10, n_compare)):
    #     print(f"  ligne {i:4d}: |ΔX[0]|={abs(diff_X[i,0]):.3e}  |ΔY[0]|={abs(diff_Y[i,0]):.3e}")

except FileNotFoundError:
    print(f"\nFichier de reference introuvable : {REF_FILE}")
    print("Simulation executee sans comparaison.")

# ── sauvegarde ────────────────────────────────────────────────────────────────
np.savetxt("data_lib.dat", state, fmt="%.15g")
print(f"\nSauvegarde : data_lib.dat  ({N_rows} lignes x {2*N} colonnes)")
