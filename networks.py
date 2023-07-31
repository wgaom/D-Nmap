import numpy as np
import scipy.linalg
from numpy.linalg import inv as inv_mat

from sympy.matrices import Matrix, zeros
from math import acos, asin, atan, sqrt, cos, sin, tan

class network:
    """
    Networks
    
    p_list: boundary points
    DN: Matrix form of Dirichlet-to-Neumann map with respect to p_list.
    """
    def __init__(self, length=0.1, end_point=['a','b'], density = 1, DN=None, pt_list=None):
        if DN is not None:
            self.DN=DN
            if pt_list is not None:
                self.p_list=pt_list
            else:
                dim, _ = np.shape(DN)
                self.p_list=[i for i in range(dim)]
            return
        self.density = density
        self.generate_line_nets(length, end_point)
        
    def generate_line_nets(self, l=0.1, endpoint=['a','b']):
        self.p_list = endpoint[:]
        self.DN = np.array([[cos(l)/sin(l),-1/sin(l)],[-1/sin(l),cos(l)/sin(l)]])
    
    def find_DN_map(self, p1, p2):
        i1 = self.p_list.index(p1)
        i2 = self.p_list.index(p2)
        return self.DN[i1,i2]
    def find_boundary_point_index(self, p):
        if p in self.p_list:
            return self.p_list.index(p)
        else:
            return -1

class V_space:
    """
    define of V1
    
    dim: dimension of V.
    rank: dimension of V1
    rank_constrain: basis of V1 in V.
    
    """
    def __str__(self) -> str:
        return f"Vspace, class, dim={self.dim}, rank={self.rank}, rank_cons={self.rank_constrain}"
    def __repr__(self) -> str:
        return self.__str__()

    def __init__(self, orientation, density = None):
        self.dim = len(orientation)
        if density == None:
            density = [1]*self.dim
        if isinstance(density, list):
            self.density = np.array(density)
        else:
            self.density = density
        self.generate_projection(np.array([orientation]),density)

    """
    General projection of inner product.
    Suppose A(x,y)=x.TB.TBy
    Given V, want to find P such that A(P(x),y)=A(x,P(y))
    and P^2=P.

    Details. Suppose Q be projection of Rn to BV,
    Then P = (B^{-1}QB)
    """

    def generate_projection(self, V:np.ndarray, density):
        """
        V, 2-dim array,
        density, 1-dim array.
        """
        sqrt_density = np.sqrt(density)
        b = np.diag(sqrt_density)
        self.B=b
        self.B_basis_C = scipy.linalg.orth(b@V.T).T

        self.rank_constrain, _ = np.shape(self.B_basis_C)
        self.rank = self.dim - self.rank_constrain
        self.B_basis = scipy.linalg.null_space(self.B_basis_C).T
        self.P = np.linalg.inv(b)@self.B_basis.T@self.B_basis@b
        self.P_C = np.linalg.inv(b)@self.B_basis_C.T@self.B_basis_C@b
    
    def regular_x(self, x):
        """
        convert x into np.array type
        """
        sp = np.shape(x)
        if len(sp)==1:
            x = np.array([x]).T
        else:
            m,n=sp
            if m==1:
                x = x.T
        return x
    def proj(self,x ,complement = False):
        """
        Return 1*n 2-d array.
        """
        x = self.regular_x(x)
        if complement:
            mat = self.P
        else:
            mat = self.P_C
        return (mat@x).T
    
    def coordinate(self, x, complement = False):
        """
        coordinate of a vector under basis of V1
        """
        x = self.regular_x(x)
        try:
            x = self.B@x
        except:
            raise Exception('value error')

        if complement:
            basis = self.B_basis_C
        else:
            basis = self.B_basis
        return (basis@x).T

class networksWithPartition:
    """
    A set of networks. 
    with identification.
    
    net_list: [N1, N2, ...], a list of networks
    point_list: all the boundary points of Ni
    int_point: interior points
    bdy_point: boundary points
    
    identification:
    [{
        N_list:[N1,N2,N3,...],
        p_list:[p1,p2,p3,...],
        V: V_space,
        offset: 0
    },
    ...
    ], a list of dictionary. Each dictionary contains several networks Ni and one boundary point on each Ni. It means we will glue those points into a large network. V_space means the V_1 space we described in our paper.
    
    The matrix is coming from indentification. So
    We need to record the index of metrix.
    """
    
    def __init__(self, graph = None, N_dict=None):
        if graph is not None:
            self.generate_from_graph(graph)
            return
        self.net_list=N_dict['N']
        for i in range(len(self.net_list)):
            pt_list = N_dict['renameEndPoint'][i]
            for j in range(len(pt_list)):
                self.net_list[i].p_list[j]=pt_list[j]
        self.generateBoundaryPts()
        self.generate_ind()

    def generateBoundaryPts(self):
        pt_list = {}
        for N in self.net_list:
            for p in N.p_list:
                if p in pt_list:
                    pt_list[p]+=1
                else:
                    pt_list[p]=1
        self.bdy_point=[]
        self.int_point=[]
        for p in pt_list:
            if pt_list[p]==1:
                self.bdy_point.append(p)
            else:
                self.int_point.append(p)
    
    def convert2nets(self):
        self.compute_boundary_DN()
        endpoint=[i for _,i in self.bdy_DN_point_list]
        return network(DN=self.bdy_DN_map,pt_list=endpoint)
        
    def generate_ind(self):
        self.ind=[]
        offset=0
        for p in self.int_point:
            nets_list=[]
            p_list=[]
            ori_list=[]
            density_list=[]
            for i in range(len(self.net_list)):
                net = self.net_list[i]
                ind_p=net.find_boundary_point_index(p)
                if ind_p>=0:
                    nets_list.append(i)
                    p_list.append(p)
                    ori_list.append(1)
                    density_list.append(1)
            V=V_space(ori_list)
            self.ind.append({
                'nets_list':nets_list,
                'p_list':p_list,
                'V': V,
                'offset':offset
            })
            offset+=V.rank

    def generate_from_edge_list(self, edge_list):
        self.net_list = []
        self.p_list = []
        self.int_point = []
        for edge in edge_list:
            if isinstance(edge[2],tuple):
                self.net_list.append(network(edge[2][0], end_point=[edge[0],edge[1]], density=edge[2][1]))
            else:
                self.net_list.append(network(edge[2], end_point=[edge[0],edge[1]]))
            for i in range(2):
                p = edge[i]
                if p not in self.p_list:
                    self.p_list.append(p)
                elif p not in self.int_point:
                    self.int_point.append(p)
                    
        self.bdy_point = []
        for p in self.p_list:
            if p not in self.int_point:
                self.bdy_point.append(p)
        
        self.ind = []
        offset = 0
        for p in self.int_point:
            nets_list = []
            p_list = []
            ori_list = []
            density_list = []
            for i in range(len(self.net_list)):
                net = self.net_list[i]
                ind_p=net.find_boundary_point_index(p)
                if ind_p>= 0:
                    nets_list.append(i)
                    p_list.append(p)
                    oritation = (-1)**ind_p
                    ori_list.append(oritation)
                    density_list.append(self.net_list[i].density)
            V = V_space(ori_list)
            self.ind.append({
                'nets_list':nets_list,
                'p_list':p_list,
                'V': V,
                'offset':offset
            })
            offset+=V.rank
        
    def find_index(self, net_index, point):
        for i in range(len(self.ind)):
            if net_index in self.ind[i]['nets_list']:
                p_index = self.ind[i]['nets_list'].index(net_index)
                if self.ind[i]['p_list'][p_index]==point:
                    return i, p_index
        return None
    
    def contract_matrix(self, mat, ind_list, P, Q):
        """
        contract ind_list,
        provided constrained contidion Qx=0,
        such that P(y)=0
        """
        rev_ind = []
        dim, _ = np.shape(mat)
        n2, _ = np.shape(Q)

        for i in range(dim):
            if i not in ind_list:
                rev_ind.append(i)
        m = len(rev_ind)
        A = mat[np.ix_(rev_ind, rev_ind)]
        B = mat[np.ix_(rev_ind, ind_list)]
        C = mat[np.ix_(ind_list,ind_list)]

        new_mat = A+B@np.linalg.inv(np.concatenate((P@C,Q),
        axis=0))@np.concatenate((-P@B.T, np.zeros((n2, m))))
        return new_mat

    def compute_boundary_DN(self):
        mat_axis = []
        dim = 0
        for i in range(len(self.net_list)):
            for p in self.net_list[i].p_list:
                mat_axis.append((i, p))
            dim += len(self.net_list[i].p_list)
        
        mat = np.zeros((dim,dim))
        
        pre_dim = 0
        for i in range(len(self.net_list)):
            dim = len(self.net_list[i].p_list)
            mat[pre_dim:pre_dim+dim, pre_dim:pre_dim+dim]=\
                self.net_list[i].DN
            pre_dim+=dim
        
        for i in range(len(self.ind)):
            p_contract_list = [] #index of point need to contract
            ind = self.ind[i]
            for j in range(len(ind['nets_list'])):
                net_index = ind['nets_list'][j]
                p = ind['p_list'][j]
                p_contract_list.append((net_index,
                p))
            P = ind['V'].B_basis @ ind['V'].B
            Q = ind['V'].B_basis_C @ ind['V'].B
            p_index_list =[]
            for p in p_contract_list:
                p_index_list.append(mat_axis.index(p))
            mat = self.contract_matrix(mat, 
                p_index_list, P, Q)
            for p in p_contract_list:
                mat_axis.remove(p)
        self.bdy_DN_map = mat
        self.bdy_DN_point_list = mat_axis
            


    def compute_DN_map(self):
        self.T_dim = 0
        for ind in self.ind:
            self.T_dim+=ind['V'].rank
        
        self.DN_map = np.zeros((self.T_dim, self.T_dim))
        
        for ind in self.ind:
            V = ind['V']
            n = V.rank # n=2
            for i in range(n):
                v = (V.B_basis@np.linalg.inv(V.B.T))[i]
                # v = [-1,1,0]/sqrt(2) e.g.
                for j in range(len(v)): #len(v) is just len(nets_list)
                    net_index = ind['nets_list'][j]
                    point = ind['p_list'][j]
                    net = self.net_list[net_index]
                    
                    for e_point in net.p_list:
                        if e_point in self.bdy_point:
                            continue
                        index, p_index = self.find_index(net_index, e_point)
                        m=self.ind[index]['V'].dim
                        arr = [0]*m
                        arr[p_index]=net.find_DN_map(point, e_point)*v[j]
                        # pdb.set_trace()
                        coor = self.ind[index]['V'].coordinate(arr)
                        _, n = np.shape(coor)
                        self.DN_map[ind['offset']+i, self.ind[index]['offset']:self.ind[index]['offset']+
                                   n]+=coor[0]
    def print_eig(self):
        """
        print eigen values
        """
        eig=np.linalg.eigh(self.DN_map)[0]
        print(f'Eigenvalues of Dirichlet-to-Neumann map is given by {eig}')
        # print(eig)
        pos=0
        for e in eig:
            if e>1e-8:
                pos+=1
        print(f'It has {pos} positive eigenvalues')

        
    def generate_from_graph(self, graph):
        edge_list = []
        for l in graph:
            for tup in graph[l]:
                for i in range(len(tup)-1):
                    edge_list.append([tup[i],tup[i+1],l])
        self.generate_from_edge_list(edge_list)