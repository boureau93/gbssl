 
typedef struct
{
    int index; //Index of node in a graph
    double *theta; //field
    double *b; // marginal probability (belief)
} node;

double gsl_exp(double x);

node init_node(int q);
void set_node_belief(node *n, int q, double *h);
void reset_node_belief(node *n, int q);
int node_mode(node n, int q);
void free_node(node *n);
