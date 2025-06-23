from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r') as f:
        data = f.read()
    
    data = data.replace('&', '&amp;')
    
    with open('copy.wa', 'w') as f:
        f.write(data)
    
    tree = ET.parse('copy.wa')
    root = tree.getroot()
    
    sentence_pairs = []
    alignments = []
    
    for s_elem in root.findall('s'):
        english_elem = s_elem.find('english')
        czech_elem = s_elem.find('czech')
        source = english_elem.text.split()
        target = czech_elem.text.split()
        sentence_pairs.append(SentencePair(source, target))
        
        sure_pairs = []
        sure_elem = s_elem.find('sure')
        if sure_elem.text and sure_elem.text.strip():
            for pair in sure_elem.text.strip().split():
                s, t = map(int, pair.split('-'))
                sure_pairs.append((s, t))
        
        possible_pairs = []
        possible_elem = s_elem.find('possible')
        if possible_elem.text and possible_elem.text.strip():
            for pair in possible_elem.text.strip().split():
                s, t = map(int, pair.split('-'))
                possible_pairs.append((s, t))
        
        alignments.append(LabeledAlignment(sure_pairs, possible_pairs))
    
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counts = defaultdict(int)
    target_counts = defaultdict(int)
    
    for pair in sentence_pairs:
        for token in pair.source:
            source_counts[token] += 1
        for token in pair.target:
            target_counts[token] += 1
    
    sorted_source = sorted(source_counts.items(), key=lambda x: (-x[1], x[0]))
    sorted_target = sorted(target_counts.items(), key=lambda x: (-x[1], x[0]))
    
    if freq_cutoff is not None:
        sorted_source = sorted_source[:freq_cutoff]
        sorted_target = sorted_target[:freq_cutoff]
    
    source_dict = {token: i for i, (token, _) in enumerate(sorted_source)}
    target_dict = {token: i for i, (token, _) in enumerate(sorted_target)}
    
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_pairs = []
    
    for pair in sentence_pairs:
        source_indices = [source_dict[token] for token in pair.source if token in source_dict]
        target_indices = [target_dict[token] for token in pair.target if token in target_dict]
        
        if source_indices and target_indices:
            tokenized_pair = TokenizedSentencePair(
                source_tokens=np.array(source_indices, dtype=np.int32),
                target_tokens=np.array(target_indices, dtype=np.int32)
            )
            tokenized_pairs.append(tokenized_pair)
    
    return tokenized_pairs
