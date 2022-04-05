cdef extern from "gsl/gsl_math.h":
    pass

cdef extern from "omp.h":
    pass

#-------------------------------------------------------------------------------
# GRAPH FUNCTIONS
#-------------------------------------------------------------------------------

cdef extern from "nodes.h":

    ctypedef struct node:
        int index #Index of node in a graph
        double *theta #field
        double *b #marginal probability (belief)
    
    int node_mode(node n, int q)

cdef extern from "pairs.h":

    ctypedef struct edge:
        node *nd1, *nd2 #nodes of current pair
        double W #Interaction strength ("similarity" or "weigth of an edge")

    edge init_edge(double W_12, double W_21, node *nd1, node *nd2)

cdef extern from "graph.h" nogil:
    
    ctypedef struct graph_model:
        int q #Number of states for each node
        node *nd #Array of nodes
        int N_nodes #Number of nodes
        edge *ed #Array of pairs
        int N_edges #Number of pairs

    graph_model init_graph_model(int q, int Nd, int Ne)
    void reset_graph_beliefs(graph_model *G)
    void free_graph_model(graph_model *G)
    
cdef extern from "potts.h" nogil:

    void nmf_propagation(graph_model *G, double beta, int t_max, double eps)
    double log_prob_mode(graph_model G)

cdef extern from "ssl.h" nogil:

    void grf(graph_model *G, int t_max, double eps)
    void lgc(graph_model *G, double alpha, int t_max, double eps)
