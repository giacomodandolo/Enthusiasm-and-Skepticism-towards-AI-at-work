import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import spacy
from spacymoji import Emoji
import emoji
import igraph as ig
import leidenalg
import scipy.sparse as sps
from scipy.sparse import diags
from bertopic import BERTopic

class SemanticNetwork():
    def __init__(self, filter_words=None):
        # Spacy pipeline
        self.filter_words = filter_words or []
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer.token_match = re.compile("^#\w+$").match
        self.nlp.add_pipe("emoji", first=True)

    def _get_emoji_regexp(self):
        # Sort emoji by length to make sure multi-character emojis are
        # matched first
        emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
        pattern = '(' + '|'.join(re.escape(u) for u in emojis) + ')'
        return re.compile(pattern)

    def _clean_text_fun(self, text, to_keep=True,
               kr_list = ["ADJ","ADV","EMOJI","HASH","NOUN","PROPN","VERB"]):

        # apply spacy
        doc = self.nlp(" ".join(self._get_emoji_regexp().split(text)))
        # collect spacy outomes in in a list
        out = list()
        for token in doc:
            out.append((token.text, token.text+token.whitespace_,
                        token.lemma_.lower(), token.pos_,
                        token.text.startswith("#"),
                        token.text.startswith("@"),
                        emoji.purely_emoji(token.text),
                        (token.text.startswith("https:") or token.text.startswith("http:"))
                        ))
        # turn it into a dataframe
        df2 = pd.DataFrame(out)
        df2.columns = ['text', 'full_text', 'lemma', 'PoS',
                      'is_hashtag', 'is_mention', 'is_emoji', 'is_link']
        # build a PoS column that identifies hashtags, mentions, etc
        df2.insert(loc = 4, column = 'myPoS', value = df2['PoS'])
        df2.loc[df2['is_hashtag']==True,'myPoS'] = 'HASH'
        df2.loc[df2['is_mention']==True,'myPoS'] = 'MENT'
        df2.loc[df2['is_emoji']==True,'myPoS'] = 'EMOJI'
        df2.loc[df2['is_link']==True,'myPoS'] = 'HTML'
        # replace emojis lemmas with their description
        for tmp in doc._.emoji:
          df2.loc[tmp[1],'lemma'] = tmp[0]+" "+tmp[2]
        # keep/remove only what is asked
        if to_keep:
          out = [j for (i,j) in zip(df2['myPoS'],df2['lemma']) if i in kr_list]
        else:
          out = [j for (i,j) in zip(df2['myPoS'],df2['lemma']) if i not in kr_list]
        return out
    
    def clean_text(self, df, content_column="translated", to_keep=True,
               kr_list = ["ADJ","NOUN","VERB"]):

        self.df = df.copy()
        self.df['clean list'] = self.df[content_column].apply(
            lambda x: self._clean_text_fun(x, to_keep, kr_list)
        )

    def get_Nwd(self, n_min=2, n_max=1e10):
        # capture execution time
        tic = time.time()

        # collection of (unique) words
        clean_texts_list = list(self.df['clean list'])
        words = np.unique([item for sublist in clean_texts_list for item in sublist])
        Nw = len(words)  # number of words
        Nd = len(clean_texts_list)  # number of documents
        documents = np.array(range(Nd))

        # occurrence matrix for words in documents
        words_dict = dict(zip(words, range(Nw)))  # word -> row index
        Nwd = sps.lil_matrix((Nw, Nd), dtype=np.int16)
        for i in range(Nd):
            for j in clean_texts_list[i]:
                Nwd[words_dict[j], i] += 1
        Nwd = Nwd.tocsr()

        # identify words used less than n_min or more than n_max
        word_counts = np.sum(Nwd, axis=1).A1  # convert (Nw,1) -> 1D array
        select = (word_counts >= n_min) & (word_counts < n_max)

        # explicitly print the most frequent words
        print('removing words...')
        with np.printoptions(threshold=np.inf):
            print(words[word_counts >= n_max])

        # remove them
        Nwd = Nwd[select, :]
        words = words[select]

        # remove documents that do not contain any words
        doc_select = np.sum(Nwd, axis=0).A1 > 0
        Nwd = Nwd[:, doc_select]
        documents = documents[doc_select]

        # capture execution time
        print(f'Occurrence matrix: execution time {time.time() - tic:.2f} [s]')

        self.Nwd = Nwd
        self.words = words
        self.documents = documents

        # plot words occurrences
        plt.figure(figsize=(4, 3))
        plt.semilogy(-np.sort(-np.asarray(np.sum(Nwd, axis=1)).reshape(-1)))
        plt.grid(True)
        plt.xlabel('word id')
        plt.ylabel('# of occurrences')
        plt.title("words occurrences")
        plt.show()

        # build other matrices
        Pwd = Nwd / Nwd.sum(axis=0).A1  # normalize by document
        print("After initial normalization:", Pwd.shape)

        # Apply weight = log(1 + |upvotes|)
        upvotes = self.df.loc[self.documents, 'upvotes'].values
        weights = np.log1p(np.abs(upvotes))
        W = diags(weights)
        Pwd = Pwd.dot(W)
        print("After applying log(upvotes) weights:", Pwd.shape)

        # normalize again
        total_weight = weights.sum()
        Pwd = Pwd / total_weight
        print("After final normalization:", Pwd.shape)

        # words and document matrices
        pd = Pwd.sum(axis=0).A1
        Pww = (Pwd / pd).dot(Pwd.T)
        pw = Pwd.sum(axis=1).A1
        Pdd = (Pwd.T / pw).dot(Pwd)

        self.Pwd = Pwd
        self.Pdd = Pdd
        self.Pww = Pww
        self.pd = pd

    def build_word_graph(self, threshold=0.0, damping=0.85, simplify=True):
        # Safety check
        if not hasattr(self, "Pww"):
            raise ValueError("Pww not found. Run get_Nwd() first.")

        # Convert Pww to sparse matrix
        A = csr_matrix(self.Pww)

        # Optional thresholding
        if threshold > 0:
            A = A.multiply(A > threshold)

        # Build adjacency (binary)
        # Obtain coordinates (i,j) of non-zero elements
        rows, cols = A.nonzero()
        weights = A[rows, cols].A1

        # Build ig graph
        G = ig.Graph()
        G.add_vertices(A.shape[0])
        edges = list(zip(rows, cols))
        G.add_edges(edges)
        G.es["weight"] = weights
        G.vs["name"] = list(self.words)

        if simplify:
            G.simplify(combine_edges="sum", loops=True)

        self.G_word = G

        # Compute PageRank
        pr = self.G_word.pagerank(
            damping=damping,
            weights="weight"
        )

        self.word_pagerank = np.array(pr)
        self.word_pagerank_dict = dict(zip(self.words, pr))

    def build_document_graph(self, threshold=0.0, simplify=True):
        # Safety check
        if not hasattr(self, "Pdd"):
            raise ValueError("Pdd not found. Run get_Pdd() first.")

        # Convert Pdd to sparse matrix
        A = csr_matrix(self.Pdd)

        # Optional thresholding
        if threshold > 0:
            A = A.multiply(A > threshold)

        # Build adjacency (binary)
        # Obtain coordinates (i,j) of non-zero elements
        rows, cols = A.nonzero()
        weights = A[rows, cols].A1 

        # Build ig graph
        G = ig.Graph()
        G.add_vertices(A.shape[0])
        edges = list(zip(rows, cols))
        G.add_edges(edges)
        G.es["weight"] = weights

        # Add node labels (documents)
        if hasattr(self, "documents"):
            G.vs["name"] = list(self.documents)
        else:
            G.vs["name"] = [f"doc_{i}" for i in range(A.shape[0])]

        # Optional cleaning
        if simplify:
            G.simplify(combine_edges="sum", loops=True)

        self.G_doc = G

    def detect_communities(self, weights="weight"):
        # Safety check
        if not hasattr(self, "G_doc"):
            raise ValueError("Document graph not found. Run build_document_graph() first.")
        G = self.G_doc
        # Run Leiden
        partition = leidenalg.find_partition(
            G,
            leidenalg.ModularityVertexPartition,
            weights=weights,
        )
        # Store partition
        G.vs["community"] = partition.membership
        self.partition = partition
        # map the partition into a community assignment matrix C
        n_docs = self.Pdd.shape[0]
        n_coms = len(partition)

        C = sps.lil_matrix((n_docs, n_coms))
        for i in range(n_coms):
            C[np.array(partition[i]), i] = 1
        C = C.tocsr()

        # builds topic matrices
        Pwc = self.Pwd.dot(C) # joint word + class probability
        Pcc = ((C.T).dot(self.Pdd)).dot(C) # joint class + class probability
        pc = Pcc.sum(axis=0)

        # Filter isolated communities
        pc_arr = np.array(pc).flatten()
        keep = pc_arr > 0

        if not np.all(keep):
            C   = C[:, keep]
            Pwc = Pwc[:, keep]
            Pcc = Pcc[keep][:, keep]
            pc  = pc[:, keep]

        self.louvain_C   = C
        self.louvain_Pwc = Pwc
        self.louvain_Pcc = Pcc
        self.louvain_pc  = pc

    def detect_communities_overlapping(self, n_iter=10, weights="weight"):
        # Safety check
        if not hasattr(self, "G_doc"):
            raise ValueError("Document graph not found. Run build_document_graph() first.")

        G = self.G_doc
        n_docs = self.Pdd.shape[0]

        # store memberships from each iteration
        all_memberships = []

        for it in range(n_iter):
            partition = leidenalg.find_partition(
                G,
                leidenalg.ModularityVertexPartition,
                weights=weights,
                seed=np.random.randint(0, 10000)
            )
            all_memberships.append(partition.membership)

        # build overlapping C matrix
        # each unique (doc, community) pair across iterations is included
        community_map = {}
        community_counter = 0

        for membership in all_memberships:
            for doc_id, com_id in enumerate(membership):
                key = (doc_id, com_id)
                if key not in community_map:
                    community_map[key] = community_counter
                    community_counter += 1

        n_coms = len(community_map)
        C = sps.lil_matrix((n_docs, n_coms), dtype=int)

        for (doc_id, com_id), col_id in community_map.items():
            C[doc_id, col_id] = 1

        C = C.tocsr()
        
        # builds topic matrices
        Pwc = self.Pwd.dot(C)
        Pcc = ((C.T).dot(self.Pdd)).dot(C)
        pc = Pcc.sum(axis=0)

        # filter isolated communities
        keep = np.array(pc).flatten() > 0
        if not np.all(keep):
            C = C[:, keep]
            Pwc = Pwc[:, keep]
            Pcc = Pcc[keep][:, keep]
            pc = pc[:, keep]

        self.louvain_C_overlapping = C
        self.louvain_Pwc_overlapping = Pwc
        self.louvain_Pcc_overlapping = Pcc
        self.louvain_pc_overlapping = pc


    def run_bertopic(self, content_column="translated", nr_topics="auto", reduce_outliers=True):
        if not hasattr(self, "df"):
            raise ValueError("Dataframe not found. Run clean_text() first.")

        # Convert lists to strings if needed
        docs = (
        self.df.loc[self.documents, content_column]
        .apply(lambda x: " ".join(x) if isinstance(x, list) else x)
        .tolist()
        )

        self.topic_model = BERTopic(nr_topics=nr_topics)
        topics, probs = self.topic_model.fit_transform(docs)

        if reduce_outliers:
            try:
                new_topics = self.topic_model.reduce_outliers(docs, topics)
                self.topic_model.update_topics(docs, topics=new_topics)
                topics = new_topics
            except ValueError as e:
                # Ignore the error
                if "No outliers to reduce" in str(e):
                    print("No outliers to reduce, proceeding.")
                    new_topics = topics
                else:
                    raise e
        else:
            new_topics = topics

        # Store results in the class
        self.bertopic_topics = topics
        self.bertopic_probs = probs
        self.bertopic_info = self.topic_model.get_topic_info()

        b_topics = np.array(new_topics)
        n_docs = len(b_topics)
        n_topics_final = b_topics.max() + 1

        C = sps.lil_matrix((n_docs, n_topics_final))
        for i in range(n_docs):
            if b_topics[i] >= 0:
                C[i, b_topics[i]] = 1
        self.BERTopic_C = C.tocsr()

        # builds topic matrices
        self.BERTopic_Pwc = self.Pwd.dot(self.BERTopic_C) # joint word + class probability
        self.BERTopic_Pcc = ((self.BERTopic_C.T).dot(self.Pdd)).dot(self.BERTopic_C) # joint class + class probability
        self.BERTopic_pc = self.BERTopic_Pcc.sum(axis=0)