# ================================================================
# TP3 - Inteligencia Artificial
# Autor: Hernán Ricardo Noya
# ================================================================

import numpy as np
import random

N = 100  # 10x10 = 100 neuronas

# --- Conversión entre matriz e imagen ---
def grid_to_vec(grid):
    vec = []
    for row in grid:
        for ch in row:
            vec.append(1 if ch in ['#', '+'] else -1)
    return np.array(vec, dtype=float)

def vec_to_grid(v):
    out = []
    for r in range(10):
        row = ''.join('#' if v[r*10+c] > 0 else '.' for c in range(10))
        out.append(row)
    return out

def print_grid(grid):
    for row in grid:
        print(row)

# --- Entrenamiento ---
def train_hebb(patterns):
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W

def train_pseudoinverse(patterns):
    S = np.stack(patterns, axis=1)
    W = S @ np.linalg.pinv(S.T @ S) @ S.T
    np.fill_diagonal(W, 0)
    return W

# --- Recuperación ---
def recall_async(x, W, max_iters=20, seed=None):
    rng = random.Random(seed)
    s = x.copy()
    for _ in range(max_iters):
        changed = False
        idx = list(range(N))
        rng.shuffle(idx)
        for i in idx:
            h = float(W[i, :] @ s)
            new_val = 1 if h > 0 else -1
            if new_val != s[i]:
                s[i] = new_val
                changed = True
        if not changed:
            break
    return s

# --- Métricas ---
def hamming_distance(a, b):
    return np.sum(a != b)

def centroid_from_grid(grid):
    xs, ys = [], []
    for r in range(10):
        for c in range(10):
            if grid[r][c] == '#':
                xs.append(c)
                ys.append(r)
    if not xs:
        return None, None
    return sum(xs)/len(xs), sum(ys)/len(ys)

# --- Patrones base ---
PATTERN_BASE = [
"..........",
"..........",
".....#....",
"....#.#...",
"...#...#..",
"..#.....#.",
"...#...#..",
"....#.#...",
"+....#....",
"+++.......",
]

def shift_pattern(grid, dr, dc):
    new = [['.' for _ in range(10)] for __ in range(10)]
    for r in range(10):
        for c in range(10):
            ch = grid[r][c]
            if ch != '.':
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    new[nr][nc] = ch
    return ["".join(row) for row in new]

def add_noise_to_grid(grid, n_flip=3, seed=None):
    rng = random.Random(seed)
    mutable = [list(row) for row in grid]
    flips = 0
    while flips < n_flip:
        r, c = rng.randrange(0,10), rng.randrange(0,10)
        if mutable[r][c] == '+':  # no tocar escuadra
            continue
        mutable[r][c] = '#' if mutable[r][c] == '.' else '.'
        flips += 1
    return ["".join(row) for row in mutable]

# --- Ejecución principal ---
def main():
    print("=== Prototipo Hopfield 10x10 ===\n")
    
    # Creo 4 patrones del aro desplazado
    base = PATTERN_BASE
    shifts = [(0,0), (0,1), (1,0), (-1,0)]
    patterns_grids = [shift_pattern(base, dr, dc) for (dr,dc) in shifts]
    patterns_vecs = [grid_to_vec(g) for g in patterns_grids]

    # Entrenamiento
    W = train_pseudoinverse(patterns_vecs)
    print("Entrenamiento con regla de pseudoinversa\n")

    # Imagen de prueba (con ruido)
    test_index = 1
    original = patterns_grids[test_index]
    noisy = add_noise_to_grid(original, n_flip=4, seed=42)

    print("Imagen original:")
    print_grid(original); print()
    print("Imagen con ruido:")
    print_grid(noisy); print()

    # Recuperación
    x = grid_to_vec(noisy)
    y = recall_async(x, W, max_iters=30, seed=42)
    rec = vec_to_grid(y)

    print("Imagen recuperada:")
    print_grid(rec); print()

    # Resultados
    cx, cy = centroid_from_grid(rec)
    if cx is not None:
        print(f"Centroide estimado: x={cx:.2f}, y={cy:.2f}\n")

    dist = [hamming_distance(y, p) for p in patterns_vecs]
    winner = int(np.argmin(dist))
    print(f"Patrón ganador: P{winner} (distancia: {dist[winner]})")
    print("=== Fin de la ejecución ===")

if __name__ == "__main__":
    main()
