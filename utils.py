import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import svdvals
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import MaxNLocator

def svd(A, full_matrices=True):
    U, s, VT = jnp.linalg.svd(A, full_matrices=full_matrices)
    return U, s, VT.T

def compose(f, g):
    def h(x):
        return f(g(x))
    return h

def compose_cls(f, g):
    def h(x, y):
        return f(g(x, y))
    return h

def compute_angle(U, V):
    if len(U.shape) < 2:
        U = U.reshape(-1, 1)
    if len(V.shape) < 2:
        V = V.reshape(-1, 1)

    s = jnp.clip(svdvals(jnp.dot(U.T, V)), 0, 1)
    theta = jnp.arccos(s)

    return jnp.max(theta)

def compute_svd_series(weights, layer, rank):
    input_dim = weights[0][0].shape[1]
    m = input_dim - 2 * rank
    
    Uinit, _, Vinit = svd(weights[1][layer])

    right_series = [[ *(input_dim * [0]) ]]
    left_series = [[ *(input_dim * [0]) ]]
    sval_series = [svdvals(weights[0][0])]

    for w in tqdm(weights[1:]):
        Ut, st, Vt = svd(w[layer])
        sval_series.append(st)

        Ut_top = Ut[:, :rank]
        Ut_mid = Ut[:, rank:-rank]
        Ut_bot = Ut[:, -rank:]

        Uinit_top = Uinit[:, :rank]
        Uinit_mid = Uinit[:, rank:-rank]
        Uinit_bot = Uinit[:, -rank:]

        Vt_top = Vt[:, :rank]
        Vt_mid = Vt[:, rank:-rank]
        Vt_bot = Vt[:, -rank:]

        Vinit_top = Vinit[:, :rank]
        Vinit_mid = Vinit[:, rank:-rank]
        Vinit_bot = Vinit[:, -rank:]

        right = []

        for k in range(rank):
            right.append(compute_angle(Vt_top[:, k], Vinit_top[:, k]))

        # right += rank * [compute_angle(Vt_top, Vinit_top)]

        # for k in range(m):
        #     right.append(compute_angle(Vt_mid[:, k], Vinit_mid[:, k]))

        right += m * [compute_angle(Vt_mid, Vinit_mid)]

        for k in range(rank):
            right.append(compute_angle(Vt_bot[:, k], Vinit_bot[:, k]))

        # right += rank * [compute_angle(Vt_bot, Vinit_bot)]

        right_series.append(right)

        left = []

        for k in range(rank):
            left.append(compute_angle(Ut_top[:, k], Uinit_top[:, k]))

        # left += rank * [compute_angle(Ut_top, Uinit_top)]

        # for k in range(m):
        #     left.append(compute_angle(Ut_mid[:, k], Uinit_mid[:, k]))

        left += m * [compute_angle(Ut_mid, Uinit_mid)]

        for k in range(rank):
            left.append(compute_angle(Ut_bot[:, k], Uinit_bot[:, k]))

        # left += rank * [compute_angle(Ut_bot, Uinit_bot)]

        left_series.append(left)

    sval_series = jnp.array(sval_series)
    right_series = jnp.array(right_series)
    left_series = jnp.array(left_series)
    
    return (sval_series, right_series, left_series)

def plot_sv_series(ax, series, color='viridis', spec_step=2, sample_step=1):
    # subsample the series 
    series = series[:, ::sample_step]

    n_time_indices, n_sval_indices = series.shape
    time_indices = jnp.arange(n_time_indices)
    sval_indices = jnp.arange(n_sval_indices)

    spectrum_verts = []

    for idx in time_indices[::spec_step]:
        spectrum_verts.append([
            (0, jnp.min(series)-0.05), *zip(sval_indices, series[idx, :]), (n_sval_indices, jnp.min(series)-0.05)
        ])

    path_verts = []

    for idx in sval_indices:
        path_verts.append([
            *zip(time_indices, series[:, idx])
        ])

    spectrum_poly = PolyCollection(spectrum_verts)
    spectrum_poly.set_alpha(0.8)
    spectrum_poly.set_facecolor(plt.colormaps[color](jnp.linspace(0, 0.7, len(spectrum_verts))))
    spectrum_poly.set_edgecolor('black')

    path_line = LineCollection(path_verts)
    path_line.set_linewidth(1)
    path_line.set_edgecolor('black')
    
    ax.set_box_aspect(aspect=None, zoom=0.85)

    ax.add_collection3d(spectrum_poly, zs=time_indices[::spec_step], zdir='y')
    ax.add_collection3d(path_line, zs=sval_indices, zdir='x')

    ax.set_xlim(0, n_sval_indices)
    ax.set_ylim(0, n_time_indices)
    ax.set_zlim(jnp.min(series)-0.1, jnp.max(series)+0.1)

    elev = 30
    azim = -50
    roll = 0
    ax.view_init(elev, azim, roll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))