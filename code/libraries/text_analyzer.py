import pandas as pd
import spacy
from spacymoji import Emoji
import emoji
import re
from tqdm import tqdm

class TextAnalyzer():
    COL_KEYWORD = "keyword"
    COL_POS = "pos"
    COL_COUNT = "count"
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

            # Filter based on pos (if available)
            if pos_filter is not None and pos not in pos_filter:
                continue

            all_tokens.append((lemma, pos))

        tmp_df = pd.DataFrame(all_tokens, columns=[self.COL_KEYWORD, self.COL_POS])
        output_df = (
            tmp_df
            .value_counts()
            .reset_index(name=self.COL_COUNT)
            .sort_values(self.COL_COUNT, ascending=False)
            .reset_index(drop=True)
        )
        return output_df
        
    # Make a dataframe that contains, for each word inside the document:
    # - Keyword text
    # - Keyword PoS
    # - Keyword frequency (count)
    def analyze_dataframe(self, data_df, content_column, pos_filter=None, keyword_filter=None, count_threshold=1):
        all_tokens = []   # (lemma, pos)
        texts = data_df[content_column].astype(str).tolist()

        # Set PoS filter
        if pos_filter is not None:
            if isinstance(pos_filter, str):
                pos_filter = {pos_filter}
            else:
                pos_filter = set(pos_filter)

        # Set keyword filter
        if keyword_filter is not None:
            if isinstance(keyword_filter, str):
                keyword_filter = {keyword_filter.lower()}
            else:
                keyword_filter = set([kw.lower() for kw in keyword_filter])

        # Extract PoS
        for doc in self.nlp.pipe(tqdm(texts, desc="Processing texts"), batch_size=64, n_process=1):
            for token in doc:
                # skip spaces, punctuation
                if token.is_space or token.is_punct:
                    continue

                lemma = token.lemma_.lower().strip()
                pos = token.pos_

                # Skip empty lemmas
                if not lemma:
                    continue

                # Filter based on keyword (if available)
                if keyword_filter is not None and lemma not in keyword_filter:
                    continue

                # Filter based on pos (if available)
                if pos_filter is not None and pos not in pos_filter:
                    continue

                all_tokens.append((lemma, pos))

        # Keyword count and dataframe creation
        tmp_df = pd.DataFrame(all_tokens, columns=[self.COL_KEYWORD, self.COL_POS])
        output_df = (
            tmp_df
            .value_counts()
            .reset_index(name=self.COL_COUNT)
            .sort_values(self.COL_COUNT, ascending=False)
            .reset_index(drop=True)
        )
        # Filter based on the count threshold
        if count_threshold > 1:
            output_df = output_df[output_df[self.COL_COUNT] > count_threshold].reset_index(drop=True)
        
        return output_df
    
    def analyze_csv(self, data_path, data_sep, content_column, pos_filter=None, keyword_filter=None, count_threshold=1):
        data_df = pd.read_csv(data_path, delimiter=data_sep)
        return self.analyze_dataframe(
            data_df=data_df,
            content_column=content_column,
            pos_filter=pos_filter,
            keyword_filter=keyword_filter,
            count_threshold=count_threshold
        )
    
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