
typedef struct 
{
    node *nd1, *nd2; //nodes of current pair
    double W_12; //Weight of the edge from nd1 to nd2
    double W_21; //Weitht of the edge from nd1 do nd2
} edge;

edge init_edge(double W_12, double W_21, node *nd1, node *nd2);
void free_edge(edge *p);
