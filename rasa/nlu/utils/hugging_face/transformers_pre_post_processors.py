from typing import List, Tuple, Text
import numpy as np


def bert_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add BERT style special tokens(CLS and SEP)"""
    BERT_CLS_ID = 101
    BERT_SEP_ID = 102

    processed_tokens = token_ids

    processed_tokens.insert(0, BERT_CLS_ID)
    processed_tokens.append(BERT_SEP_ID)

    return processed_tokens


def gpt_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    return token_ids


def xlnet_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add XLNET style special tokens"""
    XLNET_CLS_ID = 3
    XLNET_SEP_ID = 4

    token_ids.append(XLNET_SEP_ID)
    token_ids.append(XLNET_CLS_ID)

    return token_ids


def roberta_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add RoBERTa style special tokens"""
    ROBERTA_BEG_ID = 0
    ROBERTA_END_ID = 2

    token_ids.insert(0, ROBERTA_BEG_ID)
    token_ids.append(ROBERTA_END_ID)

    return token_ids


def xlm_tokens_pre_processor(token_ids: List[int]) -> List[int]:
    """Add RoBERTa style special tokens"""
    XLM_SEP_ID = 1

    token_ids.insert(0, XLM_SEP_ID)
    token_ids.append(XLM_SEP_ID)

    return token_ids


def bert_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from BERT

    by removing CLS and SEP embeddings and returning CLS token embedding as
    sentence representation"""
    sentence_embedding = sequence_embeddings[0]
    post_processed_embedding = sequence_embeddings[1:-1]

    return sentence_embedding, post_processed_embedding


def gpt_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from GPT models

    by taking a mean over sequence embeddings and returning that as sentence
    representation"""
    sentence_embedding = np.mean(sequence_embeddings, axis=0)
    post_processed_embedding = sequence_embeddings

    return sentence_embedding, post_processed_embedding


def xlnet_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from XLNet models

    by taking a mean over sequence embeddings and returning that as sentence
    representation. Remove last two time steps corresponding
    to special tokens from the sequence embeddings."""
    post_processed_embedding = sequence_embeddings[:-2]
    sentence_embedding = np.mean(post_processed_embedding, axis=0)

    return sentence_embedding, post_processed_embedding


def roberta_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from Roberta models

    by taking a mean over sequence embeddings and returning that as sentence
    representation. Remove first and last time steps
    corresponding to special tokens from the sequence embeddings."""

    post_processed_embedding = sequence_embeddings[1:-1]
    sentence_embedding = np.mean(post_processed_embedding, axis=0)

    return sentence_embedding, post_processed_embedding


def xlm_embeddings_post_processor(
    sequence_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post process embeddings from XLM models

    by taking a mean over sequence embeddings and returning that as sentence
    representation. Remove first and last time steps
    corresponding to special tokens from the sequence embeddings."""
    post_processed_embedding = sequence_embeddings[1:-1]
    sentence_embedding = np.mean(post_processed_embedding, axis=0)

    return sentence_embedding, post_processed_embedding


def bert_tokens_cleaner(token_strings: List[Text]) -> List[Text]:
    """Clean up tokens with the extra delimiters(##) BERT adds while breaking a token
    into sub-tokens"""
    tokens = [string.replace("##", "") for string in token_strings]
    return [string for string in tokens if string]


def openaigpt_tokens_cleaner(token_strings: List[Text]) -> List[Text]:
    """Clean up tokens with the extra delimiters(</w>) OpenAIGPT adds while breaking a
    token into sub-tokens"""
    tokens = [string.replace("</w>", "") for string in token_strings]
    return [string for string in tokens if string]


def gpt2_tokens_cleaner(token_strings: List[Text]) -> List[Text]:
    """Clean up tokens with the extra delimiters(</w>) GPT2 adds while breaking a token
    into sub-tokens"""
    tokens = [string.replace("Ġ", "") for string in token_strings]
    return [string for string in tokens if string]


def xlnet_tokens_cleaner(token_strings: List[Text]) -> List[Text]:
    """Clean up tokens with the extra delimiters(▁) XLNet adds while breaking a token
    into sub-tokens"""
    tokens = [string.replace("▁", "") for string in token_strings]
    return [string for string in tokens if string]
