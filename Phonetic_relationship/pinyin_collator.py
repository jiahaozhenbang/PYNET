# import random
# import warnings
# from dataclasses import dataclass
# from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


# from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
# from transformers.data.data_collator import DataCollatorMixin

# InputDataClass = NewType("InputDataClass", Any)

# """
# A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
# of PyTorch/TensorFlow tensors or NumPy arrays.
# """
# DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import tokenizers
from transformers import DataCollatorForLanguageModeling
import numpy as np
from pypinyin import pinyin,Style
from tokenize_P import is_chinese_char

def convert_list_to_pinyin(input:List):
    """
        convert token list to pinyin list if the token is chinese
    """
    output=[]
    for token in input:
        if len(token)==1:
            if is_chinese_char(ord(token)):
                output.append('['+pinyin(token,style=Style.NORMAL)[0][0]+']')
                continue
        output.append(token)
    return output

@dataclass
class DataCollatorForLanguageModeling_pinyin(DataCollatorForLanguageModeling):

    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[indices_replaced]=torch.from_numpy(np.array(self.tokenizer.convert_tokens_to_ids(
            convert_list_to_pinyin(
                self.tokenizer.convert_ids_to_tokens(
                    inputs[indices_replaced]))))).to(torch.long)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

