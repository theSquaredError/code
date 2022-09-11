from pyvis.network import Network
import networkx as nx
import random
# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# nx_graph.nodes[3]['group'] = 10
# nx_graph.add_node(20, size=20, title='couple', group=2)
# nx_graph.add_node(21, size=15, title='couple', group=2)
# nx_graph.add_edge(20, 21, weight=5)
# nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
# nt = Network('500px', '500px')
# nt.from_nx(nx_graph)
# nt.show('nx.html')


def graphVisualisation(locations):
    net = Network()
    # adding nodes to the network
    set1 = []
    set2 = []
    for id, location in enumerate(locations):
        label =str(location[0].item())+','+str(location[1].item())
        net.add_node(id+1, label=label)
        set2.append(id+1)
    node1 = random.choice(set2)
    set1.append(node1)
    set2.remove(node1)
    while len(set2)!=0:
        node2 = random.choice(set2)
        
        # adding the edge in the graph
        net.add_edge(node1,node2)

        set1.append(node2)
        set2.remove(node2)
        node1 = random.choice(set1)
    
    # adding some more edges in the graph
    for i in range(5):
        node1 = random.choice(set1)
        node2 = random.choice(set1)
        if node1!=node2:
            net.add_edge(node1,node2)

    # Visualising the graph
    net.toggle_physics(True)
    net.show('mygraph.html')

    
    

    