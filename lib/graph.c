#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "nodes.h"
#include "pairs.h"
#include "graph.h"

graph_model init_graph_model(int q, int Nd, int Ne)
/*
    Returns a graph_model structure with initialized nodes.
    q: number of states for each node
    N_nd: number of nodes
*/
{
    //Model to be returned - initialization of integer parameters
    graph_model G; G.q = q;
    G.N_nodes = Nd; G.N_edges = Ne;
    //Memory allocation for nodes and edges
    G.nd = malloc(Nd*sizeof(node));
    G.ed = malloc(Ne*sizeof(edge));

    //Parallel initialization of nodes
    #pragma omp parallel for shared(G)
    for (int i=0; i<Nd; i++)
    {
        G.nd[i] = init_node(q);
        G.nd[i].index = i;
    }

    return G;
}

void reset_graph_beliefs(graph_model *G)
/*
    Resets graph beliefs to equiprobable configuration.
*/
{   
    //Iterate over nodes
    #pragma omp parallel for shared(G)
    for (int i=0; i<G->N_nodes; i++)
    {
        reset_node_belief(&G->nd[i],G->q);
    }
}

void free_graph_model(graph_model *G)
/*
    Free memory in a graph_model structure.
*/
{   
    //Free nodes
    #pragma omp parallel for shared(G)
    for (int i=0; i<G->N_nodes; i++)
    {
        free_node(&G->nd[i]);
    }

    //Free edges
    #pragma omp parallel for shared(G)
    for (int i=0; i<G->N_edges; i++)
    {
        free_edge(&G->ed[i]);
    }

    //Free allocated memory
    free(G->nd); G->nd = NULL;
    free(G->ed); G->ed = NULL;
    G = NULL;
}