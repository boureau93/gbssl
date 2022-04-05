#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "nodes.h"

//------------------------------------------------------------------------------
//Mathematical functions
//------------------------------------------------------------------------------
double gsl_exp(double x)
/*
    Exponential function.
*/
{
    return 1+gsl_expm1(x);
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

node init_node(int q)
/*
    Initializes a node structure.
    q: number of states for the node
*/
{
    //node to be  initialized and returned
    node n;
    //Memory allocation
    n.theta = malloc(q*sizeof(double));
    n.b = malloc(q*sizeof(double));
    //Initialize instances of n
    for (int s=0; s<q; s++)
    {
        n.theta[s] = 0.;
        n.b[s] = 1./(double)q;
    }

    return n;
}

void set_node_belief(node *n, int q, double *h)
/*
    Sets belief of a node via an auxiliary field h
    n: node to be set
    q: number of states of n
    h: auxiliary field
*/
{
    //auxiliary variable for normalization
    double sum = 0.;
    //Set unormalized beliefs
    for (int s=0; s<q; s++)
    {
        n->b[s] = gsl_exp(h[s]);
        sum += n->b[s];
    }
    //Normalize belief
    for (int s=0; s<q; s++)
    {
        n->b[s] /= sum;
    }
}

void reset_node_belief(node *n, int q)
/*
    Resets belief of a node to equiprobable configuration.
    n: node to be reset
    q: number of states for n
*/
{
    for (int s=0; s<q; s++)
    {
        n->b[s] = 1./(double)q;
    }
}

void set_node_field(node *n, double *theta, int q)
/*
    Set field of a node.
    n: node
    theta: field array
    q: length of teta
*/
{
    for (int s=0; s<q; s++)
    {
        n->theta[s] = theta[s];
    }
}

int node_mode(node n, int q)
/*
    Returns the most probable state of a node
*/
{
    int mode = 0; //mode variable
    
    //Iterate over states
    for (int s=0; s<q; s++)
    {
        if(n.b[s]>n.b[mode])
        {
            mode = s;
        }
    }

    return mode;
}

void free_node(node *n)
/*
    Free memory in a node.
*/
{
    free(n->b); n->b = NULL;
    free(n->theta); n->theta = NULL;
    n = NULL;
}