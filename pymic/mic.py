# mic.py
#
# main file for computing MICe/MICr score for a variable pair
import numpy as np
import partition


def trans_ranges(ranges):
    """
    Transpose x,y range definitions.
    """
    if ranges == None: return ranges
    ranges_trans = {}
    ranges_trans["xmin"] = ranges["ymin"]
    ranges_trans["xmax"] = ranges["ymax"]
    ranges_trans["ymin"] = ranges["xmin"]
    ranges_trans["ymax"] = ranges["xmax"]
    return ranges_trans

    
def truncated_geometric_table(n, eps):
    """
    Builds truncated geometric mechanism probability table
    for all true counts f in [0, ..., n]
    with eps-DP parameter eps.

    See Ghosh+2012, Example 2.2
    """

    # convert eps to alpha
    alpha = np.power(np.e, -1 * eps)
    
    plist = np.zeros((n+1, n+1))
    
    for f in range(0, n+1):
        plist[f][0] = np.power(alpha, f) / (1 + alpha)
        plist[f][n] = np.power(alpha, n - f) / (1 + alpha)

        for i in range(1, n):
            delta = i - f
            plist[f][i] = ((1-alpha)/(1+alpha)) * np.power(alpha, np.abs(delta))

    return plist


def noisy_cells(ptable=None, cells=None, n=None):
    """
    Perturb cell counts using truncated geometric mechanism.

    ptable is pre-computed probability table
    """

    # noisy_count  = np.random.choice(range(0, n+1), p=plist)
    # return noisy_count
    # add noisy counts to each cell in cells .

    rows, cols = cells.shape
    new_cells = np.zeros(cells.shape)

    for i in range(rows):
        for j in range(cols):
            f = cells[i, j]
            probs = ptable[f]
            noisy_f = np.random.choice(range(0, n+1), p=probs)
            new_cells[i, j] = noisy_f

    # DEBUG
    # print(new_cells)
            
    return new_cells


def mice_laplace(val, n, alpha, eps):
    """
    Laplace noise mechanism for MICe statistic.
    """

    n2a = np.power(n, alpha)

    # old sensitivity (lower bound)
    # mice_sens = 1 - (2*n-n2a)/(2*n)*np.log2((2*n-n2a)/n)
    # mice_sens += n2a/(2*n)*np.log2(n/n2a)

    # new sensitivity (upper bound)
    mice_sens = n2a * ((4.8/n) + (2/n)*np.log2(n))
    
    scale = mice_sens/eps
    noise = np.random.laplace(loc=0, scale=scale)
    noisy_val = val + noise

    # truncate to 0, 1
    if noisy_val < 0: noisy_val = 0
    if noisy_val > 1: noisy_val = 1

    return noisy_val    


def micr_laplace(val, n, eps):
    """
    Laplace noise mechanism for MICr statistic.
    """
    micr_sens = (6/n) + 4*np.log2(n)/n
    scale = micr_sens/eps
    noise =  np.random.laplace(loc=0, scale=scale)
    noisy_val = val + noise

    # truncate to 0, 1
    if noisy_val < 0: noisy_val = 0
    if noisy_val > 1: noisy_val = 1

    return noisy_val


def mic_val(D=None, B=None, alpha=None, c=1, variant="mass", ranges=None,
            private=None, eps=1, ptable=None):
    """
    Compute MIC val on dataset D

    Inputs: 

    - variant: "mass" == MICe and "range" == MICr
    - ranges: for MICr variant
    - alpha: maximum number of grid cells parameter --> B = n^alpha
    - private: private mechanism type
    - eps: privacy parameter
    - ptable: precomuputed truncated geometric probability table
              using parameters eps and n = len(D)

    Output: 

    - (index, entry): of largest score in equichar matrix

    """
    n = len(D)

    if alpha: B = np.power(n, alpha)

    # compute equichar_matrix
    # if variant == "range" and private == "geometric",
    # then noisy cell counts performed in equichar_matrix routine.
    M_scores = equichar_matrix(
        D=D,
        B=B,
        c=c,
        variant=variant,
        ranges=ranges,
        private=private,
        ptable=ptable
    )

    index_max = None
    entry_max = np.max(M_scores)
    
    # privatize
    if private == "laplace":
        n = len(D)
        val = entry_max
        if variant == "mass":
            entry_max = mice_laplace(val, n, alpha, eps)
        if variant == "range":
            entry_max = micr_laplace(val, n, eps)

    return (index_max, entry_max)


def equichar_matrix(D=None, B=None, c=None, variant="mass", ranges=None,
                    private=None, ptable=None):
    """
    Compute (k, l) entries of MICe/r equichar. matrix where kl < B

    If variant == "range" and private == "geometric", 
    then cells of master equipartition are perturbed 
    using noisy_cells() with ptable, which is
    the pre-computed truncated geometric probability table.
    
    """

    M_scores = []

    # upper half of M: k (rows) <= l (cols)
    for l in range(2, int(np.floor(B/2) + 1)):
        k = l if l <= np.sqrt(B) else int(np.floor(B/l)) # k: max row size
        
        # Dtrans = transpose_data(D) 
        Dtrans = [(b, a) for a, b in D]

        # l: fixed rows size, ck: fixed  master col size
        master_gp = partition.GridPartition(Dtrans, xsize=(c*k), ysize=l)
        master_gp.equipartition(variant=variant, ranges=trans_ranges(ranges))

        # print("MASTER PARTITION:")
        # print(master_gp.joint_counts)
        
        # privatize with noisy joint_counts 
        if variant == "range" and private == "geometric":
            # print("using geometric")
            master_gp.joint_counts = noisy_cells(
                ptable=ptable,
                cells=master_gp.joint_counts,
                n=master_gp.n,
            )
        
        # find best subpartition of master_gp
        rlist = partition.OptimizeXAxis(master_gp, k)

        for entry, val in rlist:
            entry_trans = (entry[1], entry[0])
            M_scores.append(val / np.log2(entry_trans[0]))

            
    # lower half of M: k (rows) >= l (cols)
    for k in range(2, int(np.floor(B/2) + 1)):
        l = k if k <= np.sqrt(B) else int(np.floor(B/k)) # l: max col size
        
        master_gp = partition.GridPartition(D, xsize=(c*l), ysize=k)
        master_gp.equipartition(variant, ranges)

        # privatize with noisy joint_counts
        if variant == "range" and private == "geometric":
            master_gp.joint_counts = noisy_cells(
                ptable=ptable,
                cells=master_gp.joint_counts,
                n=master_gp.n,
            )

        rlist = partition.OptimizeXAxis(master_gp, l)

        for entry, val in rlist:
            M_scores.append(val / np.log2(entry[1]))
            

    return M_scores
