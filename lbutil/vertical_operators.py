import numpy as np

# HYBRID_AP and HYBRID_BP ordering: bottom up (index 0 is surface)
HYBRID_AP = np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
                      3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
                      7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
                      1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
                      2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
                      2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
                      7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01, 4.017541e+01, 3.381001e+01,
                      2.836781e+01, 2.373041e+01, 1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
                      9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00, 4.076571e+00, 3.276431e+00,
                      2.620211e+00, 2.084970e+00, 1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
                      6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
                      1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
                      1.000000e-02])
HYBRID_BP = np.array([1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                      8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                      7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                      5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                      2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                      6.960025e-03, 8.175413e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00])


def hybrid_sp(surface_pressure, ap=None, bp=None):
    if ap is None:
        ap = HYBRID_AP
    if bp is None:
        bp = HYBRID_BP
    p_edge = ap + np.atleast_1d(surface_pressure) * bp
    p_center = (p_edge[:-1] + p_edge[1:])/2
    return p_edge, p_center


def calc_regrid_weights(p_in, p_out):
    # p_out is allowed to go below (it will clip to 0); set p_out[0]=max(p_in) and p_out[1]=p_out[1] to capture surface in first box
    weights = np.zeros((len(p_out)-1, len(p_in)-1))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = (min(p_out[i], p_in[j]) - max(p_out[i+1], p_in[j+1]))/(p_in[j] - p_in[j+1])
    weights[weights < 0] = 0  # where lower lim is above the upper lim
    return weights


def regrid_to_pressure_grid(v, pressure_edges_out, total_surface_pressure_in=None, total_pressure_edges_in=None):
    # total_pressure is preferred because output pressure grid is most likely total pressure
    if total_pressure_edges_in is not None:
        pass
    elif total_surface_pressure_in is not None:
        total_pressure_edges_in, _ = hybrid_sp(total_surface_pressure_in)
    else:
        raise RuntimeError("Insufficient information to determine input pressure edges.")
    w = calc_regrid_weights(total_pressure_edges_in, pressure_edges_out)
    return w@v


def total_below(v, pressure, total_surface_pressure_in=None, total_pressure_edges_in=None, above_instead=False):
    # total_pressure is preferred because output pressure grid is most likely total pressure
    if total_pressure_edges_in is not None:
        pass
    elif total_surface_pressure_in is not None:
        total_pressure_edges_in, _ = hybrid_sp(total_surface_pressure_in)
    else:
        raise RuntimeError("Insufficient information to determine input pressure edges.")
    if above_instead:
        p_out = [pressure, total_pressure_edges_in[-1]]
    else:
        p_out = [total_pressure_edges_in[0], pressure]
    w = calc_regrid_weights(total_pressure_edges_in, p_out)
    return w@v
