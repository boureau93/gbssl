#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "nodes.h"
#include "pairs.h"
#include "graph.h"
#include "potts.h"

void nmf_propagation(graph_model *G, double beta, int t_max, double eps)
/*
    NMF propagation algorithm for a potts model.
    G: graph model
    beta: inverse temperature parameter
    t_max: maximum number of iterations
    eps: numerical precision between iterations
*/
{
    //Auxiliary field matrix initialization
    double h[G->N_nodes*G->q], aux_h[G->N_nodes*G->q];
    #pragma omp parallel for shared(h)
    for (int i=0; i<G->N_nodes*G->q; i++)
    {
        h[i] = 0.;
        aux_h[i] = 0.;
    }
    //Control variables
    int t = 0; //Number of iterations
    double delta = 1.+eps; //Numerical diff between iterations 
    
    //Propagation 
    while (t<t_max && delta>eps)
    {  

        //Iterate over edges
        #pragma omp parallel for shared(G) reduction(+:h[0:G->N_nodes*G->q])
        for (int k=0; k<G->N_edges; k++)
            //Update auxiliary fields of nodes in the k-th edge
            for (int s=0; s<G->q; s++)
            {
                h[G->q*G->ed[k].nd1->index+s] += 
                    G->ed[k].W_12*G->ed[k].nd2->b[s];
                h[G->q*G->ed[k].nd2->index+s] += 
                    G->ed[k].W_21*G->ed[k].nd1->b[s];
            }
        
        delta = 0.; //Reset delta
        //Update beliefs, reset h and calculate delta
        #pragma omp parallel for shared(G,h,delta)
        for (int i=0; i<G->N_nodes; i++)
        {
            double aux_delta = 0; //Auxiliary delta
            double sum = 0.; // Sum variable for normalization
            for (int s=0; s<G->q; s++)
            {
                //belief update
                G->nd[i].b[s] = gsl_exp(beta*(G->nd[i].theta[s]+h[G->q*i+s]));
                //sum update
                sum += G->nd[i].b[s];
                //aux_delta update
                aux_delta = gsl_max(gsl_pow_2(h[G->q*i+s]-aux_h[G->q*i+s]),
                    aux_delta);
                //aux_h update
                aux_h[G->q*i+s] = h[G->q*i+s];
                //h reset
                h[G->q*i+s] = 0.;
            }
            //Normalization
            for (int s=0; s<G->q; s++)
            {
                G->nd[i].b[s] /= sum;
            }
            //Update delta
            #pragma omp critical
            {
                if (aux_delta>delta)
                {
                    delta = aux_delta;
                }
            }
        }
        //Increase t
        t++;
    }
}

double log_prob_mode(graph_model G)
/*
    Returns the logarithm of the probability of the most probable configuration
    of a graph_model G.
*/
{
    double sum = 0.; //auxiliary variable for sum

    //Iterate over nodes
    #pragma omp parallel for shared(G) reduction(+:sum) 
    for (int i=0; i<G.N_nodes; i++)
    {
        sum += gsl_log1p(G.nd[i].b[node_mode(G.nd[i],G.q)]-1);
    }

    return sum;
}