# partition.py
#
# main file for computing MICe/MICr score for a variable pair
import numpy as np


class GridPartition:
    """
    - xsize: number of columns in P 
    - ysize: number of rows in Q

    - Xpart: x-axis partition
      --> array of size xsize, Xpart[i] == # of points in i'th part
    - Ypart: y-axis partition
      --> array of size ysize, Ypart[i] == # of points in j'th part

    - XYgrid: joint partition
      --> array of size (xsize, ysize),
          where XYgrid[i, j] == # of points in cell (i,j)


    """
    def __init__(self, D, xsize=None, ysize=None):
        self.data = D
        self.n = len(D)
        self.xsize = xsize
        self.ysize = ysize

        # initialize joint_counts array
        self.joint_counts = np.zeros((ysize,xsize), dtype=np.int64)
        # print(self.joint_counts) 

    def equipartition(self, variant, ranges):
        if variant == "mass":
            self.mass_equipartition()
        elif variant == "range":
            self.range_equipartition(ranges)

    def range_equipartition(self, ranges):
        """
        Fill in joint_counts according to range-based 
        equipartition (specified by ranges dict) of xsize rows and ysize cols

        """
        D_sorted_xy = sorted(self.data, key=lambda tup: (tup[1], tup[0]))
        D_sorted_x = sorted(self.data, key=lambda tup: tup[1])
        D_sorted_y = sorted(self.data, key=lambda tup: tup[0])

        # print(D_sorted_y)

        xmin = ranges["xmin"]; xmax = ranges["xmax"]
        xjump = (xmax - xmin)/self.xsize
        x_val_divs = [(xmin + i*xjump)  for i in range(1, self.xsize+1)]
        # x_val_divs = [i for i in np.arange(xmin + xjump, xmax, xjump)]
        x_val_divs.append(xmax + 1)
        # print(x_val_divs)

        ymin = ranges["ymin"]; ymax = ranges["ymax"]
        yjump = (ymax - ymin)/self.ysize
        y_val_divs = [(ymin + i*yjump) for i in range(1, self.ysize+1)]
        # y_val_divs = [i for i in np.arange(ymin + yjump, ymax, yjump)]
        y_val_divs.append(ymax + 1)
        
        x_id = 0
        for yval, xval in D_sorted_xy:
            y_id = 0
            if (xval >= x_val_divs[x_id]):
                x_id += 1
            
            while yval >= y_val_divs[y_id]:
                y_id += 1

            # print("{} -- xid: {}, yid: {}".format((xval, yval), x_id, y_id))
            # print(y_val_divs)
            
            if x_id >= self.xsize: x_id = self.xsize - 1
            if y_id >= self.ysize: y_id = self.ysize - 1
            
            self.joint_counts[y_id, x_id] += 1

        # print(self.joint_counts)
        
        return
            
    def mass_equipartition(self):
        """
        Fill in joint_counts according to mass-based (count-based)
        equipartition of xsize rows and ysize cols.

        """
        D_sorted_xy = sorted(self.data, key=lambda tup: (tup[1], tup[0]))
        D_sorted_x = sorted(self.data, key=lambda tup: tup[1])
        D_sorted_y = sorted(self.data, key=lambda tup: tup[0])

        # print(D_sorted_y)

        # target number of points in each column, row part
        x_target = int(np.floor(self.n/self.xsize))
        y_target = int(np.floor(self.n/self.ysize))

        # print(x_target)
        # print(y_target)

        x_val_divs = [D_sorted_x[i][1] for i in range(x_target, self.n, x_target)]
        x_val_divs.append(D_sorted_x[-1][1] + 1)
        # print(x_val_divs)
        
        y_val_divs = [D_sorted_y[i][0] for i in range(y_target, self.n, y_target)]
        y_val_divs.append(D_sorted_y[-1][0] + 1)
        # print(y_val_divs)

        x_id = 0
        for yval, xval in D_sorted_xy:
            y_id = 0
            if (xval >= x_val_divs[x_id]):
                x_id += 1
            while yval >= y_val_divs[y_id]:
                y_id += 1

            # print("{} -- xid: {}, yid: {}".format((xval, yval), x_id, y_id))
            if x_id >= self.xsize: x_id = self.xsize - 1
            if y_id >= self.ysize: y_id = self.ysize - 1
            
            self.joint_counts[y_id, x_id] += 1

        return 
                
    def row_entropy(self):
        """
        Compute entropy of the marginal row distribution
        across all points.

        """
        py_counts = []
        for i in range(self.ysize):
            count = sum(self.joint_counts[i, :])
            if count is None: continue
            py_counts.append(count)

        m = sum(py_counts)
        if m == 0: return 0

        entropy = 0
        for count in py_counts:
            py = count/m
            if py == 0: continue
            entropy += py * np.log2(py)
        entropy *= -1
                
        # DEBUG
        # print(py_counts)
        # print(m)

        return entropy
    
    def col_entropy(self, col_parts):
        """
        Compute entropy of the marginal column distribution
        given by col_parts. 

        example: 

          col_parts = [0, 1, 3, 4]
  
          Q == | p0 | p1, p2 | 

          ---> count_x1 = sum(joint_counts[:, 0:1])
          ---> count_x2 = sum(joint_counts[:, 1:3])
          ---> count_x3 = sum(joint_counts[:, 3:4])

        invariants: 
            
          - # of parts in col distribution == len(col_parts) - 1
          - last column part included in col disribution == col_parts[-1] - 1

        """
        num_parts = len(col_parts) - 1
        px_counts = []
        for i in range(num_parts):
            # count = np.sum(self.joint_counts[:, col_parts[i]:col_parts[i+1]])
            to_sum = self.joint_counts[:, col_parts[i]:col_parts[i+1]].flatten()

            # print(to_sum)
            count = 0
            for k in range(to_sum.size):
                count += to_sum[k]

            if count is None: continue
            px_counts.append(count)
        # print(px_counts)


        # m = np.sum(px_counts)
        m = 0
        for i in range(len(px_counts)):
            m += px_counts[i]
            
        # DEBUG
        # print(px_counts)
        # print(m)

        if m == 0: return 0
        
        entropy = 0
        for count in px_counts:
            px = count/m
            if px == 0: continue
            entropy += px * np.log2(px)
        entropy *= -1

        return entropy

    def joint_entropy(self, col_parts):
        """
        Compute entropy of the joint distribution 
        given by the fixed row partition and the column distribution
        given by col_parts among only those points covered by col_parts

        example: 
        
          col_parts = [0, 1, 3, 4]
        
          P = | p0 | p1, p2 | p3 | 
          Q = all Y parts (fixed)

        """
        num_col_parts = len(col_parts) - 1
        num_row_parts = self.ysize
        pxy_counts = []        
        for i in range(num_row_parts):
            for j in range(num_col_parts):
                # ct = np.sum(self.joint_counts[i,col_parts[j]:col_parts[j+1]])
                to_sum = self.joint_counts[i, col_parts[j]:col_parts[j+1]]
                ct = 0
                for k in range(to_sum.size):
                    ct += to_sum[k]
                if ct is None: continue
                pxy_counts.append(ct)

        m = 0
        for i in range(len(pxy_counts)):
            m += pxy_counts[i]

        if m == 0: return 0

        entropy = 0        
        
        for count in pxy_counts:
            pxy = count/m
            if pxy == 0: continue
            entropy += pxy * np.log2(pxy)
        entropy *= -1

        return entropy

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

    lhat = mgp.xsize
    row_entropy = mgp.row_entropy()
    
    # base cases for I, P
    for t in range(2, lhat+1):
        col_parts_max = []
        h_max = -100000000
        for s in range(0, t+1):
            col_parts = [0, s, t]
            h_val = mgp.col_entropy(col_parts) - mgp.joint_entropy(col_parts)

            if h_val > h_max:
                h_max = h_val
                col_parts_max = col_parts

        P_table[(t, 2)] = col_parts_max
        I_table[(t, 2)] = mgp.mutual_info(col_parts_max)
        # I_table[(t, 2)] = row_entropy + mgp.col_entropy(col_parts_max) - mgp.joint_entropy(col_parts_max)

    # recursive cases for I, P
    for x in range(3, l+1):
        for t in range(x, lhat+1):

            col_parts_max = []; f_max = -1000000
            for s in range(x-1, t+1):
                cs = np.sum(mgp.joint_counts[:, :s])
                ct = np.sum(mgp.joint_counts[:, :t])
            
                f_val = (cs/ct) * (I_table[(s, x-1)] - row_entropy)
                f_val -= (ct - cs)/ct * (mgp.joint_entropy([s,t]))

                if f_val > f_max:
                    f_max = f_val
                    col_parts_max = P_table[(s, x-1)].copy()
                    col_parts_max.append(t)

            P_table[(t, x)] = col_parts_max
            I_table[(t, x)] = mgp.mutual_info(col_parts_max)
            # I_table[(t, x)] = row_entropy + mgp.col_entropy(col_parts_max) - mgp.joint_entropy(col_parts_max)

    # return list of [I_(lhat,2), ..., I(lhat, l)]
    rlist = [((lhat, x), I_table[(lhat, x)]) for x in range(2, l+1)]    

    return np.array(rlist, dtype=object)

