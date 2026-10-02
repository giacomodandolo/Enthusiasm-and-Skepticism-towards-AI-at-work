import pandas as pd
import networkx as nx
import igraph as ig

class my_graph():
    def __init__(self, edges_path, nodes_path,
                 edge_sep=",", node_sep=";"):
        self.edges_df = pd.read_csv(edges_path, delimiter=edge_sep, dtype={"Weight": int})
        self.nodes_df = pd.read_csv(nodes_path, delimiter=node_sep)

        self.G_nx = nx.Graph()

        # Aggiungi nodi con attributi
        for _, row in self.nodes_df.iterrows():
            self.G_nx.add_node(row["Id"], gender=row["Gender"])

        # Aggiungi archi con pesi
        for _, row in self.edges_df.iterrows():
            self.G_nx.add_edge(row["Source"], row["Target"], weight=row["Weight"])

        self.G_ig = ig.Graph.from_networkx(self.G_nx)
        
        self.G_ig.vs["degree"] = self.G_ig.degree()
        self.G_ig.vs["pagerank"] = self.G_ig.pagerank()
        self.G_ig.vs["modularity_class"] = \
            self.G_ig.community_multilevel(weights="weight").membership

    