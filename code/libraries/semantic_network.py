import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import re
import matplotlib.pyplot as plt
from text_analyzer2_0 import TextAnalyzer

class SemanticNetwork():
    # Constants
    COL_NODE = "Node"
    COL_TYPE = "Type"
    COL_SOURCE = "Source"
    COL_TARGET = "Target"


    def __init__(
        self,
        data_path=None,
        data_sep=",",
        content_column="content",
        nodes_path=None,
        nodes_sep=",",
        edges_path=None,
        edges_sep=",",
        pos_filter=["PRON","ADJ","ADV","HASH","NOUN","PROPN","VERB"],
        min_threshold=1,
        max_threshold=None
    ):
        # Load data if a path is given
        if nodes_path is not None:
            self.nodes_df = pd.read_csv(nodes_path, delimiter=nodes_sep)
        if edges_path is not None:
            self.edges_df = pd.read_csv(edges_path, delimiter=edges_sep)

        # Generate data if a path is given
        if (nodes_path is None or edges_path is None) and data_path is not None:
            analyzer = TextAnalyzer()
            self.nodes_df, self.edges_df = analyzer.analyze_csv(
                data_path=data_path,
                data_sep=data_sep,
                content_column=content_column,
                pos_filter=pos_filter,
                min_threshold=min_threshold,
                max_threshold=max_threshold
            )  

    def build_graphs(self):
        # NetworkX graph (bipartite)
        self.G_nx = nx.Graph()

        # Add nodes with type
        for _, row in self.nodes_df.iterrows():
            self.G_nx.add_node(
                row[self.COL_NODE],
                Type=row[self.COL_TYPE]
            )

        # Add edges keyword -> comment
        for _, row in self.edges_df.iterrows():
            self.G_nx.add_edge(
                row[self.COL_SOURCE],
                row[self.COL_TARGET],
                weight=1
            )

        # Print the first 5 nodes
        print("G_nx nodes:")
        for i, (n, attr) in enumerate(self.G_nx.nodes(data=True)):
            print(n, attr)
            if i >= 4:
                break

        # Print the first 5 edges
        print("\nG_nx edges:")
        for i, (u, v, attr) in enumerate(self.G_nx.edges(data=True)):
            print(u, v, attr)
            if i >= 4:
                break

        # igraph graph (bipartite via attribute)
        nodes = list(self.G_nx.nodes())
        node2idx = {n: i for i, n in enumerate(nodes)}  # mapping nodo -> indice

        # Crea grafo vuoto con numero corretto di vertici
        self.G_ig = ig.Graph()
        self.G_ig.add_vertices(len(nodes))

        # Aggiungi attributi dei nodi
        self.G_ig.vs["name"] = nodes
        self.G_ig.vs[self.COL_TYPE] = [self.G_nx.nodes[n].get(self.COL_TYPE, "unknown") for n in nodes]

        # Converti gli edge usando gli indici
        edges_idx = [(node2idx[u], node2idx[v]) for u, v in self.G_nx.edges()]
        self.G_ig.add_edges(edges_idx)

        # Imposta attributi degli edge
        self.G_ig.es["weight"] = [self.G_nx[u][v].get("weight", 1) for u, v in self.G_nx.edges()]

        # Print the first 5 nodes
        print("G_ig nodes:")
        for v in self.G_ig.vs[:5]:
            print(f"Name: {v['name']}, Type: {v[self.COL_TYPE]}")

        # Print the first 5 edges
        print("\nG_ig edges:")
        for e in self.G_ig.es[:5]:
            source = self.G_ig.vs[e.source]["name"]
            target = self.G_ig.vs[e.target]["name"]
            weight = e["weight"]
            print(f"{source} -> {target}, weight={weight}")


    def comment_probability_vector(self, comment_id):
        # Extract keyword with an edge to the comment
        kws = (
            self.edges_df[self.edges_df[self.COL_TARGET] == comment_id]
            [self.COL_SOURCE]
        )
        # Count the number of keywords
        counts = kws.value_counts()
        #Return the normalized vector
        return counts / counts.sum()

    def build_Nwd(self):
        Nwd = (
            self.edges_df
            .assign(value=1)
            .pivot_table(
                index=self.COL_SOURCE,   # keyword
                columns=self.COL_TARGET, # comment_id
                values="value",
                aggfunc="sum",
                fill_value=0
            )
        )
        return Nwd
    
    def show_Nwd(self):
        Nwd = self.build_Nwd()
        # Plot
        plt.figure(figsize=(4, 3))
        word_occurrences = np.asarray(Nwd.sum(axis=1)).reshape(-1)
        plt.semilogy(-np.sort(-word_occurrences))
        plt.grid(True)
        plt.xlabel('word id')
        plt.ylabel('# of occurrences')
        plt.title("Words occurrences")
        plt.show()
        
    def build_Pw_given_d(self):
        Nwd = self.build_Nwd()
        P = Nwd.div(Nwd.sum(axis=0), axis=1)
        return P
    
    # Words and documents joint probability
    def build_Pwd(self):
        # Conditional probability
        Pw_given_d = self.buil_Pwd()
        # Use uniform distribution for the comments
        pd = pd.Series(
            1 / Pw_given_d.shape[1],
            index=Pw_given_d.columns
        )
        # Multiply the probabilities
        Pwd = Pw_given_d.mul(pd, axis=1)
        return Pwd

    # Words (keywords) marginal probability
    def compute_Pww(self):
            Pwd = self.build_pwd()

            # Document marginal
            Pd = Pwd.sum(axis=0)

            # Pwd^T diag(Pd)^(-1)
            left = Pwd.div(Pd, axis=1)

            # Final product
            Pww = left @ Pwd.T

            return Pww
    
    # Documents (comments) marginal probability
    def compute_Pdd(self):
        Pwd = self.build_pwd()

        # Word marginal
        Pw = Pwd.sum(axis=1)

        # Pwd^T diag(Pw)^(-1)
        left = Pwd.T.div(Pw, axis=1)

        # Final product
        Pdd = left @ Pwd

        return Pdd

    def show(self):
        fig, ax = plt.subplots(figsize=(12, 12))
        
        counts = np.array(self.G_ig.vs[self.COL_SCORE])
        
        # Normalize dimensions
        # Es. scala lineare tra 20 e 80
        min_size, max_size = 20, 80
        if len(counts) > 0:
            vertex_sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_size - min_size)
        else:
            vertex_sizes = 20  # default for empty graph

        ig.plot(
            self.G_ig,
            target=ax,
            vertex_label=self.G_ig.vs["name"],
            vertex_size=vertex_sizes,
            edge_width=2,
            edge_color="black",
            margin=50
        )
        
        plt.show()

    def to_csv(self, folder_path, file_name="network"):
        self.nodes_df.to_csv(f"{folder_path}/{file_name}_nodes.csv", index=False)
        self.edges_df.to_csv(f"{folder_path}/{file_name}_edges.csv", index=False)