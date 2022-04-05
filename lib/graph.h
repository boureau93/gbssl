
typedef struct
{
    int q; //Number of states for each node
    node *nd; //Array of nodes
    int N_nodes; //Number of nodes
    edge *ed; //Array of pairs
    int N_edges; // Number of pairs
} graph_model;

graph_model init_graph_model(int q, int Nd, int Ne);
void reset_graph_beliefs(graph_model *G);
void free_graph_model(graph_model *G);