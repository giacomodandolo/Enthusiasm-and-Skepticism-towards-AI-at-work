# This mimics BERTAgent but it can work in the absence of GPU resources,
# although slow in that case
import re
import ftfy
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

class agency():
    def __init__(self, device):
        self.device = device
        # see https://bertagent.readthedocs.io/en/latest/
        model_name = "EnchantedStardust/bertagent-best"
        revision = "5bae55efbd95dd51759d275410cea36c81109227"

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       do_lower_case = True,
                                                       revision = revision)
        # roberta model
        self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels = 1,
                revision = revision
            ).to(self.device)

    def extract_agency(self, text, BATCH_SIZE=10):

        # clean sentences (the way BERTAgent does)
        text = self._clean_sentences(text)

        # tokenize documents
        tokenized_text = self.tokenizer(
            text,
            add_special_tokens = True,
            padding = "max_length",
            truncation = True,
            max_length = 128,
            return_attention_mask = True,
            return_tensors = 'pt',
            return_token_type_ids = False,
            return_offsets_mapping = False,
        )

        # create the dataloader
        data = TensorDataset(tokenized_text['input_ids'],
                             tokenized_text['attention_mask'])
        dataloader = DataLoader(data,
                                batch_size = BATCH_SIZE)

        self.model = self.model.eval()
        out = [] # tokens + attributions

        # loop on batches
        for batch in tqdm(dataloader, desc="Evaluating"):

            # load data from dataloader
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)

            # extract embeddings
            b_embedd = self.model.roberta.embeddings.word_embeddings(
                                                   b_input_ids).detach()

            # evaluate agency
            agency = self.model.forward(inputs_embeds = b_embedd,
                                        attention_mask = b_input_mask)

            # add the result to the list
            out.append(agency['logits'].detach().cpu())

        # map the result into a dataframe
        out = np.concatenate(out)
        return out

    def _clean_sentences(self, sentences):
        sentences = [re.sub(r"\s\s+", " ", sent).strip() for sent in sentences]
        sentences = [ftfy.fix_text(sent) for sent in sentences] # fix text
        return sentences