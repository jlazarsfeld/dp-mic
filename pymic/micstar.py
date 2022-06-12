# micstar.py
#
# approximate MIC* population statistic for
# distributions (X, Y) using the
# bivariate Gaussian noise model in Reshef+16 4.4
import numpy as np
import micstar_partition as partition


def compute_boundary(mu_xvals=None, mu_yvals=None, k=None,
                     sigma=None, kmax=None, lmax=None,
                     eps_x_recip=None, eps_y_recip=None):
                     
    """
    Compute 

     - M_{up,l}: max_(P: l cols) I(X|_P, Y) / log l       

       (for 2 \le l \le lmax)

     - M_{k, up}: max_(P: k rows) I(X, Y|_P) / log k

       (for 2 \le k \le kmax

    Input: 
    - mu_xvals, mu_yvals: location of generating points
    - n: number of generating points
    - sigma: standard dev of X and Y
    - lmax: max number of cols 
    - kmax: max number of rows
    - eps_recip: size of master mass-equipartition 

    """
    # DEBUG, TODO:
    # HARD CODED: parameterize
    # eps_y_recip = 2 * eps_x_recip
    master_row_divs = np.linspace(0, 1, eps_y_recip+1)

    # compute M_{up, l}
    col_mgp = partition.GridPartition(
        mu_xvals,
        mu_yvals,
        n=k,
        sigma=sigma,
        master_x_size=eps_x_recip,
        master_y_divs=master_row_divs,
        xbounds=(0, 1),
        ybounds=(0, 1),
    )

    col_mgp.equipartition()
    col_mgp.row_partition()
    col_mgp.compute_joint_mass()

    col_ivals = partition.OptimizeXAxis(col_mgp, lmax)
    col_boundary = [e[1]/np.log2(e[0][1]) for e in col_ivals]
    col_max = np.max(col_boundary)

    # compute M_{k, up}
    row_mgp = partition.GridPartition(
        mu_yvals,
        mu_xvals,
        n=k,
        sigma=sigma,
        master_x_size=eps_x_recip,
        master_y_divs=master_row_divs,
        xbounds=(0,1),
        ybounds=(0,1)
    )
    row_mgp.equipartition()
    row_mgp.row_partition()
    row_mgp.compute_joint_mass()
    
    row_ivals = partition.OptimizeXAxis(row_mgp, kmax)
    row_boundary = [e[1]/np.log2(e[0][1]) for e in row_ivals]
    row_max = np.max(row_boundary)

    micstar_val = max(col_max, row_max)
    # micstar_val = col_max

    return (micstar_val, col_ivals, row_ivals)
