# cython: boundscheck=False, wraparound=False, nonecheck=False
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level=3

cimport decl as gdcl
from decl cimport graph_model
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

#-------------------------------------------------------------------------------
# Auxiliary functions
#-------------------------------------------------------------------------------

def adjacency_data(W):
    """
        Gets adjacency data from a square (preferably symmetric) matrix W.
    """
    Adj = [] #Array of adjacent pairs (array of edges)
    arrW = [[],[]] #Array form of W with only non-zero entries

    #Search for non-zero entries of W
    for i in range(len(W)):
        for j in range(i+1,len(W)):
            if (W[i][j]+W[j][i]!=0.):
                Adj.append((i,j))
                arrW[0].append(W[i][j])
                arrW[1].append(W[j][i])
    return Adj, arrW

#-------------------------------------------------------------------------------
# gmodel class
#-------------------------------------------------------------------------------

cdef class gmodel:

    cdef:
        public int Nd, Ne, q #Numbers of nodes, edges and states
        graph_model G #C data structure

    def __cinit__(self, cnp.ndarray W, int q):
        """
        Initialization of a graph_model.
        W: (weighted) adjacency matrix.
        q: number of states
        """
        if (W.shape[0]!=W.shape[1]):
            raise Exception("W must be a square matrix.")
        else:
            if (q<1):
                raise Exception("q must be an integer greater than 1.")
            else:
                self.Nd = W.shape[0]
                self.q = q
                #Adjacency properties
                Adj, arrW = adjacency_data(W)
                self.Ne = len(arrW[0])
                #Initialize C structure
                with nogil:
                    self.G = gdcl.init_graph_model(self.q,self.Nd,self.Ne)
                #Initialize edges
                for i in range(self.Ne):
                    self.G.ed[i] = gdcl.init_edge(arrW[0][i],arrW[1][i],
                        &self.G.nd[Adj[i][0]],&self.G.nd[Adj[i][1]])
    
    def set_fields(self, cnp.ndarray theta):
        """
            Sets fields over nodes via the theta matrix.
            theta must have shape self.Nd x self.q
        """
        if (theta.shape[0]==self.Nd and theta.shape[1]==self.q):
            for i in range(self.Nd):
                for s in range(self.q):
                    self.G.nd[i].theta[s] = theta[i][s]
        else:
            raise Exception("Shape o theta must be self.Nd x self.q")

    def get_mode(self):
        """
            Returns the most probable configuration from the marginals of
            each node.
        """
        mode = []
        for i in range(self.Nd):
            mode.append(gdcl.node_mode(self.G.nd[i],self.q))
        return mode
    
    def reset_beliefs(self):
        """
            Reset node marginals (beliefs) to equiprobable configuration.
        """
        with nogil:
            gdcl.reset_graph_beliefs(&self.G)

    #---------------------------------------------------------------------------
    # POTTS MODEL FUNCTIONS
    #---------------------------------------------------------------------------
    
    def nmf_propagation(self, double beta, int t_max, double eps):
        """
            NMF propagation algorithm for the potts model.
            beta: inverse temperature
            t_max: maximum number of iterations
            eps: numerical precision
        """
        with nogil:
            gdcl.nmf_propagation(&self.G,beta,t_max,eps)

    def log_prob_mode(self):
        """
            Returns the magnetization of a potts model calculated from beliefs.
        """
        cdef double m
        with nogil:
            m = gdcl.log_prob_mode(self.G)
        return m
    
    #---------------------------------------------------------------------------
    # GRF and LGC
    #---------------------------------------------------------------------------

    def grf(self, int t_max, double eps):
        with nogil:
            gdcl.grf(&self.G,t_max,eps)
    
    def lgc(self, double alpha, int t_max, double eps):
        with nogil:
            gdcl.lgc(&self.G,alpha,t_max,eps)
    
    #---------------------------------------------------------------------------
    # Memory deallocation
    #---------------------------------------------------------------------------

    def __dealloc__(self):
        with nogil:
            gdcl.free_graph_model(&self.G)