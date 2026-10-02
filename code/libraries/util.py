import os
import pandas as pd
import numpy as np
import igraph as ig
import scipy.sparse as sps
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import ceil, sqrt

def export_igraph_to_csv(graph, nodes_path="nodes.csv", edges_path="edges.csv"):

    if not isinstance(graph, ig.Graph):
        raise TypeError("Input must be an igraph.Graph")

    # ---------- NODES ----------
    nodes_df = pd.DataFrame({
        "id": range(graph.vcount())
    })

    # Add all vertex attributes
    for attr in graph.vs.attributes():
        nodes_df[attr] = graph.vs[attr]

    nodes_df.to_csv(nodes_path, index=False)

    # ---------- EDGES ----------
    edges_df = pd.DataFrame({
        "Source": [e.source for e in graph.es],
        "Target": [e.target for e in graph.es]
    })

    # Add all edge attributes
    for attr in graph.es.attributes():
        edges_df[attr] = graph.es[attr]

    edges_df.to_csv(edges_path, index=False)

    print(f"Saved {graph.vcount()} nodes to {nodes_path}")
    print(f"Saved {graph.ecount()} edges to {edges_path}")

def show_wordcloud(Pwc, words, pagerank, save_path=None, dpi=300, max_topics=9):

    Pwc = Pwc.toarray()
    n_topics = Pwc.shape[1]

    if max_topics is not None:
        n_topics = min(n_topics, max_topics)
    
    n_cols = int(ceil(sqrt(n_topics)))
    n_rows = int(ceil(n_topics / n_cols))

    fig, axx = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axx = np.atleast_2d(axx)

    for i in range(n_topics):
        data = Pwc[:, i]

        if data.max() == 0:
            continue

        # Combine topic relevance and PageRank
        data = data * pagerank

        if data.max() == 0:
            continue

        # Final normalization
        data = np.round(1000 * data / data.max()).astype(int)

        mask = data > 0
        wo = words[mask]
        data = data[mask]

        ax = axx[i // n_cols, i % n_cols]
        wc = WordCloud(background_color="white") \
                .generate_from_frequencies(dict(zip(wo, data)))

        ax.imshow(wc)
        ax.axis("off")
        ax.set_title(f"topic {i}")

    for j in range(i + 1, n_rows * n_cols):
        axx[j // n_cols, j % n_cols].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def save_topic_word_csv(
    Pwc,
    words,
    pagerank,
    output_dir,
    prefix="topic",
    top_k=30,
    normalize_pr=True
):
    Pwc = Pwc.toarray()
    n_topics = Pwc.shape[1]

    if isinstance(pagerank, dict):
        pr = np.array([pagerank[w] for w in words])
    else:
        pr = np.asarray(pagerank)

    if normalize_pr:
        pr = pr / pr.max()

    for t in range(n_topics):
        pwc_scores = Pwc[:, t]

        if pwc_scores.max() == 0:
            continue

        mask = pwc_scores > 0

        df = pd.DataFrame({
            "word": words[mask],
            "pagerank": pr[mask],
            "Pwc": pwc_scores[mask]
        })

        df = (
            df.sort_values("pagerank", ascending=False)
              .head(top_k)
              .reset_index(drop=True)
        )

        df["topic"] = t

        out_path = os.path.join(output_dir, f"{prefix}_{t}.csv")
        df.to_csv(out_path, index=False)

def logg(x):
    y = np.log(x)
    y[x==0] = 0
    return y

# NMI
def nmi_fn(A): # A = Pwc
    aw = A.sum(axis=1).flatten() # word probability
    ac = A.sum(axis=0).flatten() # class probability
    Hc = np.multiply(ac,-logg(ac)).sum() # class entropy
    A2 = ((A/ac).T/aw).T
    A2.data = logg(A2.data)
    y = (A.multiply(A2)).sum()/Hc
    return y

# modularity
def modularity_fn(A):
    y = A.trace()-(A.sum(axis=0)*A.sum(axis=1)).item()
    return y

# Ncut
def ncut_fn(A):
    y = ((A.sum(axis=0)-A.diagonal())/A.sum(axis=0)).mean()
    return y

# Infomap - 1
def pagerank_fn(M,q,c=.85,it=60):
    r = q.copy() # ranking matrix, initialized to q (copy)
    for k in range(it): # slow cycle
      r = c*M.dot(r) + (1-c)*q
    return r

# Infomap - 2
def _infomap_fn(v):
    y = -(v.data*logg(v.data/v.sum())).sum()
    return y

# Infomap - 3
def _infomap_entropy_fn(v):
        return -(v.data * logg(v.data / v.sum())).sum()

# Infomap - 4
def infomap_rank_fn(Pdd):
    # transition matrix
    pd = Pdd.sum(axis=0).flatten()
    M = sps.csr_matrix(Pdd/pd)
    M.data[M.data < 0] = 0
    M.eliminate_zeros()
    # pagerank vector - faster than r = pagerank_fn(M,q)
    G = ig.Graph.Adjacency((M > 0).toarray().tolist())
    G.es['weight'] = np.array(M[M.nonzero()])[0]
    r = G.pagerank(weights='weight')
    r = (sps.csr_matrix(np.array(r))).T
    return r

# Infomap - 5
def infomap_fn(C,Pdd,r):
    pd = Pdd.sum(axis=0).flatten()
    M = Pdd/pd # transition matrix
    # extract vectors
    z = (C.T).dot(sps.diags(r.toarray().flatten()))
    q = sps.csr_matrix((1,z.shape[0]))
    c = .85
    for i in range(z.shape[0]):
      tmp = ((C[:,i].transpose()).dot(M)).dot(z[i].transpose())
      q[0,i] = (1-(1-c)*C[:,i].sum()/M.shape[0])*z[i].sum()-c*tmp[0,0]
    # extract statistics
    y = _infomap_fn(q)
    for i in range(z.shape[0]):
      y += _infomap_fn(sps.hstack([z[i],sps.csr_matrix([[q[0,i]]])]))
    # normalize
    y = (y/_infomap_fn(pd))-1
    return y

def infomap_sparse(Pdd, C):
    pd = np.array(Pdd.sum(axis=0)).flatten()
    eps = 1e-12
    pd[pd == 0] = eps
    M = Pdd.multiply(1 / pd)  # Pdd / pd

    r = np.ones(Pdd.shape[0]) / Pdd.shape[0]
    c = 0.85
    for _ in range(60):
        r = c * (M.dot(r)) + (1 - c) / Pdd.shape[0]

    # z = C^T * diag(r)
    z = C.T.multiply(r[None, :])

    score = 0
    for i in range(z.shape[0]):
        col = z.getrow(i).toarray().flatten()
        col_sum = col.sum()
        if col_sum > 0:
            p = col / col_sum
            score += -np.sum(p * np.log(p + eps))

    total_entropy = -np.sum(r * np.log(r + eps))
    return score / (total_entropy + eps)

def extract_metrics(
    Pwc,
    Pcc,
    C,
    Pdd,
    pc,
    out_path,
    label=None
):

    results = {
        "NMI": nmi_fn(Pwc),
        "Q": modularity_fn(Pcc),
        "Ncut": ncut_fn(Pcc),
    }

    results["Infomap"] = infomap_sparse(Pdd, C)

    if pc.shape[1] == 1:
        results["com"] = 0
    else:
        results["com"] = _infomap_entropy_fn(pc) / np.log(pc.shape[1])

    # optional label
    if label is not None:
        results["method"] = label

    # save CSV (single row)
    df = pd.DataFrame([results])
    df.to_csv(out_path, index=False)

    return df

def save_C_to_csv(C, out_path, document_names=None, cluster_names=None):
    if sps.issparse(C):
        C_dense = C.toarray()
    else:
        C_dense = np.array(C)

    # get column names
    if cluster_names is None:
        cluster_names = [f"cluster_{i}" for i in range(C_dense.shape[1])]

    # Build dataframe
    df = pd.DataFrame(C_dense, columns=cluster_names)

    # Name of the documents
    if document_names is not None:
        df.insert(0, "document", document_names)

    # Sace to csv
    df.to_csv(out_path, index=False)