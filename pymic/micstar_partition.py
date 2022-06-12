# micstar_partition.py
#
#
# Main optimizeXaxis subroutine for approximating
# mic-star, as defined in Reshef+16 Theorem 18. 
import numpy as np
import micstar_utils as utils 
import scipy.integrate as integrate
import scipy.misc as misc
import scipy.stats as stats
import copy


class GridPartition:
    """
    Grid Partition for bivariate Gaussian noise model 
    specified in Reshef+16. 

    - mu_xvals: set of x means of generating points
    - mu_yvals: set of y means of generating points
    
    - master_x_size: size of master column partition 
    - master_y_size: size of master row partition

    - master_y_divs: row partition dividers

    """
    def __init__(self, mu_xvals, mu_yvals, n=None,
                 sigma=None, master_x_size=None, master_y_divs=None,
                 xbounds=None, ybounds=None):
        """
        Note: mu_xvals are along discrete axis,
              mu_yvals are along continuous axis
        """
        self.mu_xvals = mu_xvals
        self.mu_yvals = mu_yvals
        self.n = n
        self.sigma = sigma

        self.master_x_size = master_x_size
        self.master_y_size = len(master_y_divs) - 1
        
        self.master_xdivs = np.zeros(master_x_size + 1)
        self.master_ydivs = master_y_divs
        
        self.master_col_mass = np.zeros(self.master_x_size)
        self.master_row_mass = np.zeros(self.master_y_size)
        self.master_joint_mass = np.zeros((self.master_y_size, self.master_x_size))

        self.row_entropy_val = None

        self.xbounds = xbounds
        self.ybounds = ybounds
        self.total_joint_mass = self.compute_total_joint_mass()


    def compute_total_joint_mass(self):
        """
        Compute total mass in region specified
        by self.xbounds and self.ybounds. 
        """
        xa = self.xbounds[0]; xb = self.xbounds[1]
        ya = self.ybounds[0]; yb = self.ybounds[1]

        total_mass = utils.joint_mass(
            xa, xb, ya, yb,
            self.mu_xvals, self.mu_yvals, self.n, self.sigma
        )
        
        return total_mass
        
        
    def equipartition(self):
        """
        Mass-based equipartition of x-axis into 
        self.master_size parts.
        """

        print("equipartition X axis")
        
        # set xdiv boundaries
        self.master_xdivs[0] = self.xbounds[0]
        self.master_xdivs[-1] = self.xbounds[1]
        
        # (X, Y) params
        x_pts = self.mu_xvals
        y_pts = self.mu_yvals
        n = self.n
        sigma = self.sigma
        total_mass = self.total_joint_mass
        ya = self.ybounds[0]
        yb = self.ybounds[1]
        
        # target mass per column
        target = 1/self.master_x_size
        delta = 0.00001
        
        for i in range(1, self.master_x_size):

            # DEBUG
            # print("equipartition xdiv {}/{}".format(i+1, self.master_x_size))

            # need xb to start (tmp_low + tmp_hi) / 2
            xa = self.master_xdivs[i-1]
            tmp_low = xa if xa != -np.inf else -200
            tmp_hi = xa + 1 if xa != -np.inf else 1
            xb = (tmp_low + tmp_hi)/2

            # initial mass in [xa, xb]
            mass = utils.joint_mass(
                xa, xb, ya, yb,
                x_pts, y_pts, n, sigma
            )
            mass *= 1 / total_mass
            
            iters = 0
            while np.abs(mass - target) > delta:

                if mass > target: tmp_hi = xb
                elif mass < target: tmp_low = xb
                    
                xb = (tmp_low + tmp_hi) / 2

                mass = utils.joint_mass(
                    xa, xb, ya, yb,
                    x_pts, y_pts, n, sigma
                )
                mass *= 1 / total_mass
                iters += 1

                # DEBUG
                # s = "iters = {}, mass = {}, xb = {}"
                # print(s.format(iters, round(mass, 5), xb))

            self.master_xdivs[i] = xb

        xb_final = self.xbounds[1]
        self.master_xdivs[-1] = xb_final

        return

    
    def row_partition(self):
        """
        Compute mass of rows in master row partition, where
        rows are specified by master_y_divs. 

        Normalize wrt mass within (xbounds, ybounds).

        """
        master_ydivs = self.master_ydivs
        mu_yvals = self.mu_yvals
        mu_xvals = self.mu_xvals
        n = self.n; 
        sigma = self.sigma

        # xbounds to keep track of valid region
        xa = self.xbounds[0]
        xb = self.xbounds[1]

        total_mass = self.total_joint_mass

        for i in range(len(master_ydivs)-1):
            ya = master_ydivs[i]
            yb = master_ydivs[i+1]

            ymass_in_bounds = utils.joint_mass(
                xa, xb, ya, yb,
                mu_xvals, mu_yvals,
                n, sigma
            )
            ymass = ymass_in_bounds / total_mass
            self.master_row_mass[i] = ymass
            
        return
    
    
    def compute_joint_mass(self):
        """
        Compute joint probability mass function of 
        the (discrete) distribution specified by 
        master_y_divs and master_x_divs. 
        """
        print("computing joint mass")
        
        master_xdivs = self.master_xdivs; master_ydivs = self.master_ydivs
        mu_xvals = self.mu_xvals; mu_yvals = self.mu_yvals
        n = self.n; sigma = self.sigma

        total_cells = (len(master_xdivs)-1) * (len(master_ydivs)-1)
        count = 0

        
        # fill in joint mass table
        for j in range(len(master_xdivs)-1):

            xa = master_xdivs[j]; xb = master_xdivs[j+1]

            for i in range(len(master_ydivs)-1):
                ya = master_ydivs[i]; yb = master_ydivs[i+1]
                mass = utils.joint_mass(
                    xa, xb, ya, yb,
                    mu_xvals, mu_yvals, n, sigma,
                    total_mass=self.total_joint_mass
                )
                self.master_joint_mass[i, j] = mass

                count += 1
                # print("compute_joint {} / {}".format(count, total_cells))

        # fill in row and col marginal mass tables
        row_masses = []
        for row in self.master_joint_mass:
            row_masses.append(sum(row))

        col_masses = []
        for j in range(self.master_x_size):
            val = sum([row[j] for row in self.master_joint_mass])
            col_masses.append(val)

        self.master_row_mass = np.array(row_masses)
        self.master_col_mass = np.array(col_masses)

        return

                
    def row_entropy(self):
        """
        Compute entropy of the continuous marginal 
        row distribution.
        
        """
        H = 0
        for mass in self.master_row_mass:
            if mass == 0: continue
            H += mass * np.log2(mass)
        H *= -1
        return H                    

    
    def col_entropy(self, col_parts):
        """
        Compute discrete entropy of the marginal
        colum distribution given by col_divs.

        example:

        col_parts = [0, 1, 3, 4]
        Q == | p0 | p1, p2 | p3
        
        mass_x1 = master_col_mass[0:1][0]
        mass_x2 = master_col_mass[1:3][0]
        mass_x3 = master_col-mass[3:4][0]

        """
        num_parts = len(col_parts) - 1
        px_masses = []
        for i in range(num_parts):
            mass_parts = self.master_col_mass[col_parts[i]:col_parts[i+1]]
            mass = sum(mass_parts)
            if mass != 0: px_masses.append(mass)

        total_mass = sum(px_masses)
        px_masses_normalized = [px/total_mass for px in px_masses]
            
        H = 0
        for px in px_masses_normalized:
            H += px * np.log2(px)
        H *= -1

        return H

    
    def joint_entropy(self, col_parts):
        """
        Compute entropy of joint distribution (X, Y)
        - where X is discrete and specified by col_parts
        - Y is continuous

        example:
        
        col_parts = [0, 1, 3, 4]
        """

        num_col_parts = len(col_parts) - 1
        num_row_parts = self.master_y_size
        pxy_masses = []
        
        for i in range(num_row_parts):
            for j in range(num_col_parts):
                to_sum = self.master_joint_mass[i, col_parts[j]:col_parts[j+1]]
                mass = 0
                for k in range(to_sum.size): mass += to_sum[k]
                    
                if mass is None: continue
                pxy_masses.append(mass)

        M = 0
        for i in range(len(pxy_masses)): M += pxy_masses[i]
        if M == 0: return 0

        H = 0        
        for mass in pxy_masses:
            pxy = mass / M
            if pxy <= 0: continue # guards against small neg roundoff error case
            H += pxy * np.log2(pxy)
        H *= -1

        return H

    
    def mutual_info(self, col_parts):
        """
        Compute mutual information of the joint distribution
        given by the fixed row partition and the column distribution
        given by col_parts among only those points covered by col_parts.
        
        """
        mi = self.row_entropy()
        mi += self.col_entropy(col_parts)
        mi -= self.joint_entropy(col_parts)
        return mi

    
def OptimizeXAxis(mgp, l):
    """
    Given master GridParition mgp with
    
      - fixed y-axis row-partition Q of size 
      - master x-axis col-partition PI of size lhat
      - max x-axis column size l

    Returns the set of mutual information values

      (I_2, I_3,  .... , I_l)

    Where each I_i is the highest mutual information of
    the joint distribution (P_i, Q), where P_i is a 
    x-axis col partition of size i that is a sub-partition of PI.
    
    """

    I_table = {}
    P_table = {}

    lhat = mgp.master_x_size # size of master partition
    row_entropy = mgp.row_entropy()
    mgp.row_entropy_val = row_entropy

    # DEBUG:
    print("OptimizeXAxis: base cases")
    
    # base cases for I, P
    for t in range(2, lhat+1):

        # print("t = {} / {}".format(t, lhat+1))
        
        col_parts_max = []
        h_max = -100000000
        for s in range(0, t+1):
            col_parts = [0, s, t]
            h_val = mgp.col_entropy(col_parts) - mgp.joint_entropy(col_parts)
            # h_val = mgp.col_entropy(col_parts) - mgp.joint_entropy_OLD(col_parts)

            if h_val > h_max:
                h_max = h_val
                col_parts_max = col_parts

        P_table[(t, 2)] = col_parts_max
        I_table[(t, 2)] = mgp.mutual_info(col_parts_max)

    
    # DEBUG:
    print("OptimizeXAxis: recursive cases")

    # recursive cases for I, P
    for x in range(3, l+1):
        
        for t in range(x, lhat+1):

            # print("x = {} / {} \t t = {} / {}".format(x, l+1, t, lhat+1))

            col_parts_max = []; f_max = -1000000
            for s in range(x-1, t+1):

                cs = np.sum(mgp.master_col_mass[:s])
                ct = np.sum(mgp.master_col_mass[:t])
                            
                f_val = (cs/ct) * (I_table[(s, x-1)] - row_entropy)
                f_val -= (ct - cs)/ct * (mgp.joint_entropy([s,t]))

                if f_val > f_max:
                    f_max = f_val
                    col_parts_max = P_table[(s, x-1)].copy()
                    col_parts_max.append(t)

            P_table[(t, x)] = col_parts_max
            I_table[(t, x)] = mgp.mutual_info(col_parts_max)

    # return list of [I_(lhat,2), ..., I(lhat, l)]
    rlist = [((lhat, x), I_table[(lhat, x)], I_table[(lhat, x)]/np.log2(x)) for x in range(2, l+1)]

    return np.array(rlist, dtype=object)

