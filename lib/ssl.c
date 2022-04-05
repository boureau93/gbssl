#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "nodes.h"
#include "pairs.h"
#include "graph.h"
#include "potts.h"

void grf(graph_model *G, int t_max, double eps)
/*
    Gaussian random fields propagation algorithm.
    G: graph model
    t_max: maximum number of iterations
    eps: numerical precision
*/
{
    //Binary array of length G->N_nodes
    int labeled[G->N_nodes];
    //Auxiliary array for beliefs
    double aux_b[G->N_nodes*G->q];

    //Initialization of beliefs for labeled data and of aux_b
    #pragma omp parallel for shared(G,labeled,aux_b)
    for (int i=0; i<G->N_nodes; i++)
    {   
        //Binary variable
        int is_labeled = 0;
        //Iterate over states to check if labeled
        for (int s=0; s<G->q; s++)
        {
            //initialize aux_b
            aux_b[G->q*i+s] = 0.;
            //Check if labeled
            if (G->nd[i].theta[s]==1.)
            {
                for (int k=0; k<G->q; k++)
                    if (k==s)
                    {
                        G->nd[i].b[k] = 1.;
                    }
                    else
                    {
                        G->nd[i].b[k] = 0.;
                    }
                is_labeled = 1;
            }

        }
        labeled[i] = is_labeled;
    }

    //Control parameters for propagation
    int t=0; //number of iteratios
    double delta = 1.+eps; //numerical precision

    //Propagation dynamics
    while (t<t_max && delta>=eps)
    {
        //Iterate over edges
        #pragma omp parallel for shared(G) reduction(+:aux_b[0:G->q*G->N_nodes])
        for (int k=0; k<G->N_edges; k++)
            //Iterate over states
            for (int s=0; s<G->q; s++)
            {
                //Indexes of nodes
                int id1 = G->ed[k].nd1->index;
                int id2 = G->ed[k].nd2->index;
                //Update aux_b
                if (labeled[id1]==0)
                {
                    aux_b[G->q*id1+s] += G->ed[k].W_12*G->nd[id2].b[s];
                }
                if (labeled[id2]==0)
                {
                    aux_b[G->q*id2+s] += G->ed[k].W_21*G->nd[id1].b[s];
                }
            }
        
        //Reset delta
        delta = 0.;

        //Update beliefs. Iterate over nodes
        #pragma omp parallel for shared(G,aux_b)
        for (int i=0; i<G->N_nodes; i++)
        {
            if (labeled[i]==0)
            {
                //Auxiliary delta
                double aux_delta = 0.;
                //Iterate over states
                for (int s=0; s<G->q; s++)
                {
                    //Update aux_delta
                    aux_delta = gsl_max(gsl_pow_2(G->nd[i].b[s]-aux_b[G->q*i+s]),
                        aux_delta);
                    //Update belief
                    G->nd[i].b[s] = aux_b[G->q*i+s];
                    //Reset aux_b
                    aux_b[G->q*i+s] = 0.;
                }
                //Update delta
                #pragma omp critical
                {
                    if(aux_delta>delta)
                    {
                        delta = aux_delta;
                    }
                }
            }
        }
        //Update t
        t++;
    }
}   

void lgc(graph_model *G, double alpha, int t_max, double eps)
/*
    Local and global consistency propagation algorithm.
    G: graph model
    alpha: model parameter
    t_max: maximum number of iterations
    eps: numerical precision
*/
{
    //Auxiliary array for beliefs
    double aux_b[G->N_nodes*G->q];

    //Initialization of aux_b
    #pragma omp parallel for shared(G,aux_b)
    for (int i=0; i<G->N_nodes; i++)
        for (int s=0; s<G->q; s++)
        {
            aux_b[G->q*i+s] = (1.-alpha)*G->nd[i].theta[s];
        }

    //Control parameters for propagation
    int t=0; //number of iteratios
    double delta = 1.+eps; //numerical precision

    //Propagation dynamics
    while (t<t_max && delta>=eps)
    {
        //Iterate over edges
        #pragma omp parallel for shared(G,alpha) reduction(+:aux_b[0:G->q*G->N_nodes])
        for (int k=0; k<G->N_edges; k++)
            //Iterate over states
            for (int s=0; s<G->q; s++)
            {
                //Indexes of nodes
                int id1 = G->ed[k].nd1->index;
                int id2 = G->ed[k].nd2->index;
                //Update aux_b
                aux_b[G->q*id1+s] += alpha*G->ed[k].W_12*G->nd[id2].b[s];
                aux_b[G->q*id2+s] += alpha*G->ed[k].W_21*G->nd[id1].b[s];
            }
        
        //Reset delta
        delta = 0.;

        //Update beliefs. Iterate over nodes
        #pragma omp parallel for shared(G,aux_b)
        for (int i=0; i<G->N_nodes; i++)
        {
            //Auxiliary delta
            double aux_delta = 0.;
            //Iterate over states
            for (int s=0; s<G->q; s++)
            {
                //Update aux_delta
                aux_delta = gsl_max(gsl_pow_2(G->nd[i].b[s]-aux_b[G->q*i+s]),
                    aux_delta);
                //Update belief
                G->nd[i].b[s] = aux_b[G->q*i+s];
                //Reset aux_b
                aux_b[G->q*i+s] = (1.-alpha)*G->nd[i].theta[s];
            }
            //Update delta
            #pragma omp critical
            {
                if(aux_delta>delta)
                {
                    delta = aux_delta;
                }
            }
        }
        //Update t
        t++;
    }
}   
