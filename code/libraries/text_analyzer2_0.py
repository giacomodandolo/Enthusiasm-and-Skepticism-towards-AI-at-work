import pandas as pd
from itertools import combinations
import spacy
from spacymoji import Emoji
import emoji
import re
from tqdm import tqdm

class TextAnalyzer():
    # Constants
    COL_NODE = "Node"
    COL_TYPE = "Type"
    COL_SOURCE = "Source"
    COL_TARGET = "Target"
    COL_POS = "pos"
    COL_CLEAN = "clean"

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        self.nlp.tokenizer.token_match = re.compile("^#\w+$").match
        self.nlp.add_pipe("emoji", first=True)

    def analyze_text(self, text, pos_filter=None):
        all_tokens = []   # (lemma, pos)

        # Set filter
        if pos_filter is not None:
            if isinstance(pos_filter, str):
                pos_filter = {pos_filter}  # converto in set
            else:
                pos_filter = set(pos_filter)
        
        doc = self.nlp(text)

        for token in doc:
            # skip spaces, punctuation
            if token.is_space or token.is_punct:
                continue

            lemma = token.lemma_.lower().strip()
            pos = token.pos_

            # Skip empty lemmas
            if not lemma:
                continue

            lemma = lemma.replace("’", "'").replace("‘", "'")  # curly apostrophes → semplice
            lemma = lemma.replace("“", '"').replace("”", '"')   # virgolette curly → doppie

            # Filter based on pos (if available)
            if pos_filter is not None and pos not in pos_filter:
                continue

            all_tokens.append((lemma, pos))

        tmp_df = pd.DataFrame(all_tokens, columns=[self.COL_KEYWORD, self.COL_POS])
        nodes_df = (
            tmp_df
            .value_counts()
            .reset_index(name=self.COL_COUNT)
            .sort_values(self.COL_COUNT, ascending=False)
            .reset_index(drop=True)
        )
        return nodes_df
        
    # Make a dataframe that contains, for each word inside the document:
    # - Keyword text
    # - Keyword PoS
    # - Keyword score
    def analyze_dataframe(
        self,
        data_df,
        content_column="content",
        pos_filter=None,
        keyword_filter=None,
        min_threshold=1,
        max_threshold=None
    ):
        keyword_nodes = set()
        comment_nodes = set()
        edges = []   # (keyword, comment)

        texts = data_df[content_column].astype(str).tolist()
        indices = data_df.index.tolist()

        # PoS filter
        if pos_filter is not None:
            pos_filter = {pos_filter} if isinstance(pos_filter, str) else set(pos_filter)

        # Keyword filter
        if keyword_filter is not None:
            keyword_filter = (
                {keyword_filter.lower()}
                if isinstance(keyword_filter, str)
                else {kw.lower() for kw in keyword_filter}
            )

        # Comment analysis
        for comment_text, comment_idx, doc in zip(
            texts,
            indices,
            self.nlp.pipe(tqdm(texts, desc="Processing texts"), batch_size=64)
        ):
            comment_keywords = set()
            comment_nodes.add(comment_idx)

            for token in doc:
                if token.is_space or token.is_punct:
                    continue

                lemma = token.lemma_.lower().strip()
                pos = token.pos_

                # Filter non lemma
                if not lemma:
                    continue

                # Filter empty word and word not belonging to the keyword filter (if available)
                if keyword_filter is not None and lemma not in keyword_filter:
                    continue

                # Filter based on the PoS (if the filter is available)
                if pos_filter is not None and pos not in pos_filter:
                    continue

                comment_keywords.add(lemma)

            for kw in comment_keywords:
                keyword_nodes.add(kw)
                edges.append((kw, comment_idx))

        # Nodes dataframe
        nodes_df = pd.DataFrame(
            [(k, "keyword") for k in keyword_nodes] +
            [(c, "comment") for c in comment_nodes],
            columns=[self.COL_NODE, self.COL_TYPE]
        )

        # Edges dataframe
        edges_df = pd.DataFrame(
            edges,
            columns=[self.COL_SOURCE, self.COL_TARGET]
        )

        # Optional filtering by keyword count
        print("Keywords before filter:", len(nodes_df[nodes_df[self.COL_TYPE] == "keyword"]))
        print("Commenst before filter:", len(nodes_df[nodes_df[self.COL_TYPE] == "comment"]))
        print("Edges before filter:", len(edges_df))
        nodes_df, edges_df = self._filter_keywords_by_threshold(
            nodes_df=nodes_df,
            edges_df=edges_df,
            min_threshold=min_threshold,
            max_threshold=max_threshold
        )
        print("Keywords after filter:", len(nodes_df[nodes_df[self.COL_TYPE] == "keyword"]))
        print("Comments after filter:", len(nodes_df[nodes_df[self.COL_TYPE] == "comment"]))
        print("Edges after filter:", len(edges_df))

        nodes_df = nodes_df.reset_index(drop=True)
        edges_df = edges_df.reset_index(drop=True)

        return nodes_df, edges_df
    
    def analyze_csv(self,
                    data_path,
                    data_sep, 
                    content_column="content", 
                    pos_filter=None, 
                    keyword_filter=None, 
                    min_threshold=1, 
                    max_threshold=None):
        data_df = pd.read_csv(data_path, delimiter=data_sep)
        return self.analyze_dataframe(
            data_df=data_df,
            content_column=content_column,
            pos_filter=pos_filter,
            keyword_filter=keyword_filter,
            min_threshold=min_threshold,
            max_threshold=max_threshold
        )
    
    def _filter_keywords_by_threshold(self, nodes_df, edges_df, min_threshold=1, max_threshold=None):
        kw_counts = edges_df[self.COL_SOURCE].value_counts()
        
        # Apply minimum threshold
        valid_keywords = set(kw_counts[kw_counts >= min_threshold].index)
        
        # Apply maximum threshold
        if max_threshold is not None:
            valid_keywords = valid_keywords.intersection(set(kw_counts[kw_counts <= max_threshold].index))
        
        # Filter edges
        edges_df = edges_df[edges_df[self.COL_SOURCE].isin(valid_keywords)]
        
        # Mantain only comments that have an edge to at least keyword
        connected_comments = set(edges_df[self.COL_TARGET])
        
        # Filter nodes
        nodes_df = nodes_df[
            ((nodes_df[self.COL_TYPE] == "keyword") & (nodes_df[self.COL_NODE].isin(valid_keywords))) |
            ((nodes_df[self.COL_TYPE] == "comment") & (nodes_df[self.COL_NODE].isin(connected_comments)))
        ].reset_index(drop=True)
        
        edges_df = edges_df.reset_index(drop=True)
        
        return nodes_df, edges_df
    
    def add_clean_text(self, data_df, content_column):
        clean_texts = []
        texts = data_df[content_column].astype(str).tolist()

        for doc in self.nlp.pipe(tqdm(texts, desc="Processing texts"), batch_size=64, n_process=1):
            lemmas = []
            for token in doc:
                lemmas.append(token.lemma_.lower())

            # Substitute emoji
            for emoji_text, index, description in doc._.emoji:
                lemmas[index] = description

            clean_string = " ".join(lemmas)
            clean_texts.append(clean_string)

        data_df[f"{content_column}_{self.COL_CLEAN}"] = clean_texts

        return data_df

    def add_clean_text_from_csv(self, data_path, data_sep, content_column):
        data_df = pd.read_csv(data_path, delimiter=data_sep)
        return self.add_clean_text(
            data_df=data_df,
            content_column=content_column
        )