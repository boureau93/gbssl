#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "nodes.h"
#include "pairs.h"

edge init_edge(double W_12, double W_21, node *nd1, node *nd2)
/*
    Initializes an edge between nodes.
    q: number of states 
    W_12: weight of the edge from nd1 to nd2
    W_21: weight of the edge from nd2 to nd1
    nd1 and nd2: nodes in the pair
*/
{
    edge e;
    //Set nodes
    e.nd1 = nd1; e.nd2 = nd2;
    //Set interaction strength
    e.W_12 = W_12;
    e.W_21 = W_21;
    return e;
}

void free_edge(edge *e)
/*
    Free memory in a pair structure
*/
{
    e->nd1 = NULL; e->nd2 = NULL;
    e = NULL;
}