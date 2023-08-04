import argparse
from argparse import Namespace
import json
import os
import sys
import re
from collections import defaultdict, Counter
import copy

import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# from dsets import KnownsDataset
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util import func_register as f_reg
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally
from experiments import causal_trace_uskg as ctu

from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)

# from uskg.models.unified.prefixtuning import Model
from uskg.models.unified import finetune, prefixtuning
from uskg.utils.configue import Configure
from uskg.utils.training_arguments import WrappedSeq2SeqTrainingArguments
from uskg.seq2seq_construction import spider as s2s_spider
from uskg.third_party.spider.preprocess.get_tables import dump_db_json_schema
from uskg.third_party.spider import evaluation as sp_eval


# f_reg.register('trace_exp', '5.0')
def trace_exp5_0_dirty_attention_vector_effect(
    mt,
    a_ex,                   # output from ctu.create_analysis_sample_dicts()
    part='encoder',         # 'encoder', 'decoder'
    attn_type='self_attn',  # 'self_attn', 'cross_attn'
    corrupt_type='zero',    # 'zero', 'replace', 'add'
    window=10,              # the window size to try to corrupt
    device='cuda',
):
    """
    AAA
    Exp5 (dirty attention vector effect)
    TODO: update definition, add `attn_corrupt_type`
    """

    expect_input_ranges = a_ex['expect_input_ranges']
    tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']

    self_ranges = a_ex['self_ranges']
    context_ranges = a_ex['context_ranges']

    self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    context_tok_indices = corrupt_tok_indices = [i for s, e in context_ranges for i in range(s, e)]
    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))

    result = ctu.make_basic_result_dict(a_ex)
    result['self_ranges'] = self_ranges
    result['struct_context_ranges'] = context_ranges

    result['trace_scores'] = {
        'single_layer_corrupt': dict(),     # this layer attention vector corrupt
        'window_corrupt': dict(),        # this & neighbor (-5 to +5) layers attention vector corrupt
        'low_layers_corrupt': dict(),       # this & lower layers attention vector corrupt
        'high_layers_corrupt': dict(),      # this & higher layers attention vector corrupt
    }

    ## Check basic results
    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * 11,
        [dec_prompt] * 11,
        answer=expect,
        device=device
    )

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = a_ex['base_score']
    is_correct_pred = result['correct_prediction']
    
    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    base_score = result['base_score']

    """ Low score of exp 5.0: corrupting all layer self_toks """

    if corrupt_type == 'add':
        noise = 0.1
        replace = False
    elif corrupt_type == 'replace':
        noise = 0.1
        replace = True
    elif corrupt_type == 'zero':
        noise = 0.0
        replace = True
    
    N_layers = mt.num_enc_layers if part == 'encoder' else mt.num_dec_layers

    # low score: corrupting all layers for self toks
    low_score = result['low_score'] = ctu.trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=[],
        states_to_unpatch=[],
        answers_t=answers_t,
        states_to_corrupt=[(tnum, ctu.layername_uskg(mt.model, part, l, attn_type))
                        for tnum in self_tok_indices
                        for l in range(N_layers)],
        noise=noise,
        replace=replace,
    ).item()

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp5.0 """

    for layer_id in range(N_layers):
        # single layer
        _score = ctu.trace_with_repatch_uskg(
            model=mt.model,
            inp=inp,
            states_to_patch=[],
            states_to_unpatch=[],
            answers_t=answers_t,
            states_to_corrupt=[(tnum, ctu.layername_uskg(mt.model, part, layer_id, attn_type))
                            for tnum in self_tok_indices],
            noise=noise,
            replace=replace,
        ).item()
        result['trace_scores']['single_layer_corrupt'][layer_id] = _score

        # window
        _score = ctu.trace_with_repatch_uskg(
            model=mt.model,
            inp=inp,
            states_to_patch=[],
            states_to_unpatch=[],
            answers_t=answers_t,
            states_to_corrupt=[(tnum, ctu.layername_uskg(mt.model, part, l, attn_type))
                            for tnum in self_tok_indices
                            for l in range(
                                max(0, layer_id - window // 2),
                                min(N_layers, layer_id + window // 2)
                            )],
            noise=noise,
            replace=replace,
        ).item()
        result['trace_scores']['window_corrupt'][layer_id] = _score

        # lower layers 
        _score = ctu.trace_with_repatch_uskg(
            model=mt.model,
            inp=inp,
            states_to_patch=[],
            states_to_unpatch=[],
            answers_t=answers_t,
            states_to_corrupt=[(tnum, ctu.layername_uskg(mt.model, part, l, attn_type))
                            for tnum in self_tok_indices
                            for l in range(0, layer_id + 1)],
            noise=noise,
            replace=replace,
        ).item()
        result['trace_scores']['low_layers_corrupt'][layer_id] = _score

        # high layers 
        _score = ctu.trace_with_repatch_uskg(
            model=mt.model,
            inp=inp,
            states_to_patch=[],
            states_to_unpatch=[],
            answers_t=answers_t,
            states_to_corrupt=[(tnum, ctu.layername_uskg(mt.model, part, l, attn_type))
                            for tnum in self_tok_indices
                            for l in range(layer_id, N_layers)],
            noise=noise,
            replace=replace,
        ).item()
        result['trace_scores']['high_layers_corrupt'][layer_id] = _score

    return result


# f_reg.register('trace_exp', '5.2')
def trace_exp5_2_attention_section_removal_effect(
    mt,
    a_ex,                   # output from ctu.create_analysis_sample_dicts()
    part='encoder',         # 'encoder', 'decoder'
    attn_type='self_attn',  # 'self_attn', 'cross_attn'
    corrupt_type='zero',    # 'zero', 'replace', 'add'
    window=10,              # the window size to try to corrupt
    device='cuda',
):
    """
    AAA
    Exp5.2 (attention-section-removal-effect)
    Check the effect of disabling self-node's attention to certain sections (prefix, text, struct)
    TODO: update definition, add `attn_corrupt_type`
    """

    assert not (part == 'decoder' and attn_type == 'self_attn'), 'Decoder self-attn not implemented'

    expect_input_ranges = a_ex['expect_input_ranges']
    tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']

    self_ranges = a_ex['self_ranges']
    struct_context_ranges = a_ex['context_ranges']

    self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]
    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))
    prefix_len = mt.model.preseqlen

    result = ctu.make_basic_result_dict(a_ex)
    result['self_ranges'] = self_ranges
    result['struct_context_ranges'] = struct_context_ranges

    ## Check basic results
    n_samples = 2 if corrupt_type == 'zero' else 11     # "zero" doesn't have randomness 

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * n_samples,
        [dec_prompt] * n_samples,
        answer=expect,
        device=device
    )

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = a_ex['base_score']
    is_correct_pred = result['correct_prediction']
    
    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    base_score = result['base_score']

    """ Low score of exp 5.2: corrupting all attention sections for all layers """

    # if corrupt_type == 'add':
    #     noise = 0.1
    #     replace = False
    # elif corrupt_type == 'replace':
    #     noise = 0.1
    #     replace = True
    # elif corrupt_type == 'zero':
    #     noise = 0.0
    #     replace = True
    if corrupt_type != 'zero':
        raise NotImplementedError

    N_layers = mt.num_enc_layers if part == 'encoder' else mt.num_dec_layers

    att_sections_dict = {
        'prefix': list(range(prefix_len)),
        'text': [i + prefix_len for i in text_tok_indices],
        'self': [i + prefix_len for i in self_tok_indices],
        'struct': [i + prefix_len for i in struct_tok_indices],
        'text+struct': [i + prefix_len for i in text_tok_indices + struct_tok_indices],
        ## ADDED
        'struct_context': [i + prefix_len for i in struct_context_tok_indices],
        'text+struct_context': [i + prefix_len for i in text_tok_indices + struct_context_tok_indices],
        ## END
        'all': list(range(prefix_len)) + [i + prefix_len for i in text_tok_indices + struct_tok_indices],
    }

    # low score: corrupting all layers for self toks
    low_score = result['low_score'] = ctu.trace_attention_manip_uskg_multi_token(
        model=mt.model,
        inp=inp,
        # states_to_patch=[],
        # states_to_unpatch=[],
        answers_t=answers_t,
        layers_to_mix=[ctu.layername_uskg(mt.model, part, l, attn_type) for l in range(N_layers)],
        src_tokens_to_mix=self_tok_indices,
        tgt_tokens_to_mix=att_sections_dict['all'],
        # noise=noise,
        # replace=replace,
    ).item()

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp5.2 """

    result['trace_scores'] = dict()
    for att_section in att_sections_dict.keys():
        result['trace_scores'][att_section] = {
            'window': dict(),           # this & neighbor (-5 to +5) layers attention vector corrupt
            'first_layer': None,
            'last_layer': None,
            'all_layers': None,
        }

    for att_section, tgt_indices in att_sections_dict.items():
        ## TEMP: only run for newly added sections
        if 'struct_context' not in att_section:
            continue
        ## END TEMP
        # window
        for layer_id in range(N_layers):
            _score = ctu.trace_attention_manip_uskg_multi_token(
                model=mt.model,
                inp=inp,
                answers_t=answers_t,
                layers_to_mix=[ctu.layername_uskg(mt.model, part, l, attn_type)
                               for l in range(
                                    max(0, layer_id - window // 2),
                                    min(N_layers, layer_id + window // 2)
                                )],
                src_tokens_to_mix=self_tok_indices,
                tgt_tokens_to_mix=tgt_indices,
            ).item()
            result['trace_scores'][att_section]['window'][layer_id] = _score
        
        # single layer 
        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            layers_to_mix=[ctu.layername_uskg(mt.model, part, 0, attn_type)],
            src_tokens_to_mix=self_tok_indices,
            tgt_tokens_to_mix=tgt_indices,
        ).item()
        result['trace_scores'][att_section]['first_layer'] = _score

        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            layers_to_mix=[ctu.layername_uskg(mt.model, part, N_layers - 1, attn_type)],
            src_tokens_to_mix=self_tok_indices,
            tgt_tokens_to_mix=tgt_indices,
        ).item()
        result['trace_scores'][att_section]['last_layer'] = _score

        # all layers 
        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            layers_to_mix=[ctu.layername_uskg(mt.model, part, l, attn_type) for l in range(N_layers)],
            src_tokens_to_mix=self_tok_indices,
            tgt_tokens_to_mix=tgt_indices,
        ).item()
        result['trace_scores'][att_section]['all_layers'] = _score

    return result


@f_reg.register('trace_exp', '5.3')
def trace_exp5_3_attention_section_mutual_removal(
    mt,
    a_ex,                   # output from ctu.create_analysis_sample_dicts()
    # part='encoder',         # 'encoder', 'decoder'
    # attn_type='self_attn',  # 'self_attn', 'cross_attn'
    corrupt_type='zero',    # 'zero', 'replace', 'add'
    attn_corrupt_type='weights',    # 'weights', 'logits'
    # window=10,              # the window size to try to corrupt
    device='cuda',
):
    """
    AAA
    Exp5.3.1 (attention-section-mutual-removal)
    Check the effect of disabling attention between certain sections (prefix, text, struct)
    """

    # assert (part == 'encoder' and attn_type == 'self_attn'), 'Only Eecoder self-attn implemented'
    part = 'encoder'
    attn_type = 'self_attn'

    expect_input_ranges = a_ex['expect_input_ranges']
    tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']

    self_ranges = a_ex['self_ranges']
    struct_context_ranges = a_ex['context_ranges']

    self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]
    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))
    prefix_len = mt.model.preseqlen

    self_tok_indices_tgt_side = [i + prefix_len for i in self_tok_indices]
    struct_context_tok_indices_tgt_side = [i + prefix_len for i in struct_context_tok_indices]

    result = ctu.make_basic_result_dict(a_ex)
    # already included in make_basic_result_dict
    # result['self_ranges'] = self_ranges
    # result['struct_context_ranges'] = struct_context_ranges

    ## Check basic results
    n_samples = 2 if corrupt_type == 'zero' else 11     # "zero" doesn't have randomness 

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * n_samples,
        [dec_prompt] * n_samples,
        answer=expect,
        device=device
    )

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = a_ex['base_score']
    is_correct_pred = result['correct_prediction']

    bs, seq_len = inp['input_ids'].size()
    
    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    base_score = result['base_score']

    # """ Low score of exp 5.3: corrupting all attention between sections for all layers """
    """ Low score of exp 5.3.1: corrupting all attention weights (should be identical to exp5.0) """

    # if corrupt_type == 'add':
    #     noise = 0.1
    #     replace = False
    # elif corrupt_type == 'replace':
    #     noise = 0.1
    #     replace = True
    # elif corrupt_type == 'zero':
    #     noise = 0.0
    #     replace = True
    if corrupt_type != 'zero':
        raise NotImplementedError

    N_layers = mt.num_enc_layers if part == 'encoder' else mt.num_dec_layers

    # att_tgt_sections_dict = {
    #     'prefix': list(range(prefix_len)),
    #     'text': [i + prefix_len for i in text_tok_indices],
    #     'self': [i + prefix_len for i in self_tok_indices],
    #     'struct': [i + prefix_len for i in struct_tok_indices],
    #     'text+struct': [i + prefix_len for i in text_tok_indices + struct_tok_indices],
    #     'all': list(range(prefix_len)) + [i + prefix_len for i in text_tok_indices + struct_tok_indices],
    # }

    att_mix_mask_dict = ctu.build_enc_self_attention_mask(
        a_ex=a_ex,
        seq_len=seq_len,
        prefix_len=prefix_len,
        use_self_node=True
    )

    # corrupted_vocab_probs: (answer_len, vocab_size)
    corrupted_vocab_probs = ctu.run_attention_manip_uskg_multi_token(
        model=mt.model,
        inp=inp,
        answer_len=len(answers_t),
        mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : att_mix_mask_dict['all'] for l in range(N_layers)},
        attn_corrupt_type=attn_corrupt_type,
    )
    corrupted_answers_t = corrupted_vocab_probs.argmax(dim=-1)
    corrupted_answer = ctu.decode_sentences(mt.tokenizer, corrupted_answers_t)
    result['corrupted_answers_t'] = corrupted_answers_t.cpu().tolist()
    result['corrupted_answer'] = corrupted_answer

    # This is adapted from trace_with_repatch_uskg() (last part)
    # Compute the prob of correct answer (not corrupted answer)!
    corrupted_all_ans_probs = []
    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        # vocab_probs: [answer_len, vocab_size]
        prob = corrupted_vocab_probs[i, _t].item()
        corrupted_all_ans_probs.append(prob)

    low_score = result['low_score'] = min(corrupted_all_ans_probs)

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp5.3 """

    result['trace_scores'] = dict()
    for mix_k in att_mix_mask_dict.keys():
        result['trace_scores'][mix_k] = {
            # 'window': dict(),           # this & neighbor (-5 to +5) layers attention vector corrupt
            # 'first_layer': None,
            # 'last_layer': None,
            # 'all_layers': None,
            'low_layers': None,     # 0 - 11
            'high_layers': None,    # 12 - 23
            'all_layers': None,
        }

    for mix_k, mix_mask in att_mix_mask_dict.items():
        # TEMP: only run for newly added sections
        # if 'c' not in mix_k:
        # if mix_k != 'c->p':
        #     continue
        # END TEMP

        # # window (for speed, only compute on a subset of layers)
        # # for layer_id in range(N_layers):
        # for layer_id in range(3, N_layers, 4):
        #     _score = ctu.trace_attention_manip_uskg_multi_token(
        #         model=mt.model,
        #         inp=inp,
        #         answers_t=answers_t,
        #         mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask
        #                        for l in range(
        #                             max(0, layer_id - window // 2),
        #                             min(N_layers, layer_id + window // 2)
        #                         )},
        #     ).item()
        #     result['trace_scores'][mix_k]['window'][layer_id] = _score
        
        # # single layer 
        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, 0, attn_type) : mix_mask},
        # ).item()
        # result['trace_scores'][mix_k]['first_layer'] = _score

        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, N_layers - 1, attn_type) : mix_mask},
        # ).item()
        # result['trace_scores'][mix_k]['last_layer'] = _score

        # # all layers 
        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers)},
        # ).item()
        # result['trace_scores'][mix_k]['all_layers'] = _score

        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers // 2)},
            attn_corrupt_type=attn_corrupt_type,
        ).item()
        result['trace_scores'][mix_k]['low_layers'] = _score

        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers // 2, N_layers)},
            attn_corrupt_type=attn_corrupt_type,
        ).item()
        result['trace_scores'][mix_k]['high_layers'] = _score

        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers)},
            attn_corrupt_type=attn_corrupt_type,
        ).item()
        result['trace_scores'][mix_k]['all_layers'] = _score

    return result


@f_reg.register('trace_exp', '5.4')
def trace_exp5_4_decoder_cross_attention_removal(
    mt,
    a_ex,                   # output from ctu.create_analysis_sample_dicts()
    # part='decoder',         # 'encoder', 'decoder'
    # attn_type='cross_attn',  # 'self_attn', 'cross_attn'
    corrupt_type='zero',    # 'zero', 'replace', 'add'
    attn_corrupt_type='weights',    # 'weights', 'logits'
    # window=10,              # the window size to try to corrupt
    device='cuda',
):
    """
    AAA
    Exp5.4 (decoder-cross-attention-removal)
    Check the effect of disabling cross_attention in decoder to encoder certain sections (prefix, text, struct)
    """

    # assert (part == 'encoder' and attn_type == 'self_attn'), 'Only Eecoder self-attn implemented'
    part = 'decoder'
    attn_type = 'cross_attn'


    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    # text_range = a_ex['text_range']
    # struct_range = a_ex['struct_range']

    # self_ranges = a_ex['self_ranges']
    # struct_context_ranges = a_ex['context_ranges']
    # self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    # struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]
    # text_tok_indices = list(range(*text_range))
    # struct_tok_indices = list(range(*struct_range))
    prefix_len = mt.model.preseqlen

    # self_tok_indices_tgt_side = [i + prefix_len for i in self_tok_indices]
    # struct_context_tok_indices_tgt_side = [i + prefix_len for i in struct_context_tok_indices]

    result = ctu.make_basic_result_dict(a_ex)
    # result['self_ranges'] = self_ranges
    # result['struct_context_ranges'] = struct_context_ranges

    ## Check basic results
    n_samples = 2 if corrupt_type == 'zero' else 11     # "zero" doesn't have randomness 

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * n_samples,
        [dec_prompt] * n_samples,
        answer=expect,
        device=device
    )

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = a_ex['base_score']
    is_correct_pred = result['correct_prediction']

    bs, enc_seq_len = inp['input_ids'].size()
    bs, dec_seq_len = inp['decoder_input_ids'].size()
    
    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    base_score = result['base_score']

    """ Low score of exp 5.4: corrupting all attention weights """

    # if corrupt_type == 'add':
    #     noise = 0.1
    #     replace = False
    # elif corrupt_type == 'replace':
    #     noise = 0.1
    #     replace = True
    # elif corrupt_type == 'zero':
    #     noise = 0.0
    #     replace = True
    if corrupt_type != 'zero':
        raise NotImplementedError

    N_layers = mt.num_dec_layers

    att_mix_mask_dict = ctu.build_dec_cross_attention_mask(
        a_ex=a_ex,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        prefix_len=prefix_len,
        use_self_node=True
    )

    # low_score = result['low_score'] = ctu.trace_attention_manip_uskg_multi_token(
    #     model=mt.model,
    #     inp=inp,
    #     answers_t=answers_t,
    #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : att_mix_mask_dict['all'] for l in range(N_layers)},
    #     # layers_to_mix=[ctu.layername_uskg(mt.model, part, l, attn_type) for l in range(N_layers)],
    #     # src_tokens_to_mix=self_tok_indices,
    #     # tgt_tokens_to_mix=att_tgt_sections_dict['all'],
    #     # noise=noise,
    #     # replace=replace,
    # ).item()

    # corrupted_vocab_probs: (answer_len, vocab_size)
    corrupted_vocab_probs = ctu.run_attention_manip_uskg_multi_token(
        model=mt.model,
        inp=inp,
        answer_len=len(answers_t),
        mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : att_mix_mask_dict['all'] for l in range(N_layers)},
        attn_corrupt_type=attn_corrupt_type,
    )
    corrupted_answers_t = corrupted_vocab_probs.argmax(dim=-1)
    corrupted_answer = ctu.decode_sentences(mt.tokenizer, corrupted_answers_t)
    result['corrupted_answers_t'] = corrupted_answers_t.cpu().tolist()
    result['corrupted_answer'] = corrupted_answer

    # This is adapted from trace_with_repatch_uskg() (last part)
    # Compute the prob of correct answer (not corrupted answer)!
    corrupted_all_ans_probs = []
    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        # vocab_probs: [answer_len, vocab_size]
        prob = corrupted_vocab_probs[i, _t].item()
        corrupted_all_ans_probs.append(prob)

    low_score = result['low_score'] = min(corrupted_all_ans_probs)

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp5.4 """

    layers_range_dict = {
        # 'q1_layers': range(N_layers // 4),
        # 'q2_layers': range(N_layers // 4, N_layers // 2),
        # 'q3_layers': range(N_layers // 2, N_layers * 3 // 4),
        # 'q4_layers': range(N_layers * 3 // 4, N_layers),
        'low_layers': range(N_layers // 2),
        'mid_layers': range(N_layers // 4, N_layers * 3 // 4),
        'high_layers': range(N_layers // 2, N_layers),
        'all_layers': range(N_layers),
    }

    result['trace_scores'] = dict()
    for mix_k in att_mix_mask_dict.keys():
        result['trace_scores'][mix_k] = dict()
        # {
        #     'low_layers': None,     # 0 - 11
        #     'mid_layers': None,     # 6 - 17
        #     'high_layers': None,    # 12 - 23
        #     'all_layers': None,
        # }

    for mix_k, mix_mask in att_mix_mask_dict.items():
        # TEMP: only run for newly added sections
        # if 'c' not in mix_k:
        # if mix_k != 'c->p':
        #     continue
        # END TEMP

        for layer_k, layer_range in layers_range_dict.items():
            _score = ctu.trace_attention_manip_uskg_multi_token(
                model=mt.model,
                inp=inp,
                answers_t=answers_t,
                mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in layer_range},
                attn_corrupt_type=attn_corrupt_type,
            ).item()
            result['trace_scores'][mix_k][layer_k] = _score

        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers // 2)},
        #     attn_corrupt_type=attn_corrupt_type,
        # ).item()
        # result['trace_scores'][mix_k]['low_layers'] = _score

        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers // 2, N_layers)},
        #     attn_corrupt_type=attn_corrupt_type,
        # ).item()
        # result['trace_scores'][mix_k]['high_layers'] = _score

        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers)},
        #     attn_corrupt_type=attn_corrupt_type,
        # ).item()
        # result['trace_scores'][mix_k]['all_layers'] = _score

    return result


@f_reg.register('trace_exp', '5.5')
def trace_exp5_5(
    mt,
    a_ex,                           # output from ctu.create_analysis_sample_dicts()
    corrupt_type='zero',            # 'zero', 'replace', 'add'
    attn_corrupt_type='weights',    # 'weights', 'logits'
    device='cuda',
):
    """
    AAA
    Exp5.5 (both-part-attention-removal)
    Check the effect of disabling attention in both encoder and decoder to certain input sections (only text for now)
    """
    # part = 'decoder'
    # attn_type = 'cross_attn'

    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    # text_range = a_ex['text_range']
    # struct_range = a_ex['struct_range']

    # self_ranges = a_ex['self_ranges']
    # struct_context_ranges = a_ex['context_ranges']
    # self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    # struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]
    # text_tok_indices = list(range(*text_range))
    # struct_tok_indices = list(range(*struct_range))
    prefix_len = mt.model.preseqlen

    # self_tok_indices_tgt_side = [i + prefix_len for i in self_tok_indices]
    # struct_context_tok_indices_tgt_side = [i + prefix_len for i in struct_context_tok_indices]

    result = ctu.make_basic_result_dict(a_ex)
    # result['self_ranges'] = self_ranges
    # result['struct_context_ranges'] = struct_context_ranges

    ## Check basic results
    n_samples = 2 if corrupt_type == 'zero' else 11     # "zero" doesn't have randomness 

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * n_samples,
        [dec_prompt] * n_samples,
        answer=expect,
        device=device
    )

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = a_ex['base_score']
    is_correct_pred = result['correct_prediction']

    bs, enc_seq_len = inp['input_ids'].size()
    bs, dec_seq_len = inp['decoder_input_ids'].size()
    
    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    base_score = result['base_score']

    """ Low score of exp 5.4: corrupting all attention weights """

    # if corrupt_type == 'add':
    #     noise = 0.1
    #     replace = False
    # elif corrupt_type == 'replace':
    #     noise = 0.1
    #     replace = True
    # elif corrupt_type == 'zero':
    #     noise = 0.0
    #     replace = True
    if corrupt_type != 'zero':
        raise NotImplementedError

    N_enc_layers = mt.num_enc_layers
    N_dec_layers = mt.num_dec_layers
    assert N_enc_layers == N_dec_layers, (N_enc_layers, N_dec_layers)
    N_layers = N_enc_layers

    enc_att_mix_mask_dict = ctu.build_enc_self_attention_mask(
        a_ex=a_ex,
        seq_len=enc_seq_len,
        prefix_len=prefix_len,
        use_self_node=True
    )

    dec_att_mix_mask_dict = ctu.build_dec_cross_attention_mask(
        a_ex=a_ex,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        prefix_len=prefix_len,
        use_self_node=True
    )

    # low_score = result['low_score'] = ctu.trace_attention_manip_uskg_multi_token(
    #     model=mt.model,
    #     inp=inp,
    #     answers_t=answers_t,
    #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : att_mix_mask_dict['all'] for l in range(N_layers)},
    #     # layers_to_mix=[ctu.layername_uskg(mt.model, part, l, attn_type) for l in range(N_layers)],
    #     # src_tokens_to_mix=self_tok_indices,
    #     # tgt_tokens_to_mix=att_tgt_sections_dict['all'],
    #     # noise=noise,
    #     # replace=replace,
    # ).item()

    # corrupted_vocab_probs: (answer_len, vocab_size)
    corrupted_vocab_probs = ctu.run_attention_manip_uskg_multi_token(
        model=mt.model,
        inp=inp,
        answer_len=len(answers_t),
        mix_mask_per_layer=dict(
            [(ctu.layername_uskg(mt.model, 'encoder', l, 'self_attn'), enc_att_mix_mask_dict['all']) for l in range(N_enc_layers)] + \
            [(ctu.layername_uskg(mt.model, 'decoder', l, 'cross_attn'), dec_att_mix_mask_dict['all']) for l in range(N_dec_layers)]
        ),
        attn_corrupt_type=attn_corrupt_type,
    )
    corrupted_answers_t = corrupted_vocab_probs.argmax(dim=-1)
    corrupted_answer = ctu.decode_sentences(mt.tokenizer, corrupted_answers_t)
    result['corrupted_answers_t'] = corrupted_answers_t.cpu().tolist()
    result['corrupted_answer'] = corrupted_answer

    # This is adapted from trace_with_repatch_uskg() (last part)
    # Compute the prob of correct answer (not corrupted answer)!
    corrupted_all_ans_probs = []
    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        # vocab_probs: [answer_len, vocab_size]
        prob = corrupted_vocab_probs[i, _t].item()
        corrupted_all_ans_probs.append(prob)

    low_score = result['low_score'] = min(corrupted_all_ans_probs)

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp5.5 """

    # sect_k -> (enc_sect_k, dec_sect_k)
    section_keys_dict = {
        's->t&all->t': ('s->t', 'all->t'),
    }

    layers_range_dict = {
        'q1_layers': range(N_layers // 4),
        'q2_layers': range(N_layers // 4, N_layers // 2),
        'q3_layers': range(N_layers // 2, N_layers * 3 // 4),
        'q4_layers': range(N_layers * 3 // 4, N_layers),
        'low_layers': range(N_layers // 2),
        'mid_layers': range(N_layers // 4, N_layers * 3 // 4),
        'high_layers': range(N_layers // 2, N_layers),
        'all_layers': range(N_layers),
    }

    layer_keys_dict = {
        'E-all&D-all': ('all_layers', 'all_layers'),
        'E-all&D-low': ('all_layers', 'low_layers'),
        'E-low&D-all': ('low_layers', 'all_layers'),
    }

    result['trace_scores'] = dict()
    for sect_k in section_keys_dict.keys():
        result['trace_scores'][sect_k] = dict()

    for sect_k, (enc_sect_k, dec_sect_k) in section_keys_dict.items():
        # TEMP: only run for newly added sections
        # if 'c' not in mix_k:
        # if mix_k != 'c->p':
        #     continue
        # END TEMP

        enc_mix_mask = enc_att_mix_mask_dict[enc_sect_k]
        dec_mix_mask = dec_att_mix_mask_dict[dec_sect_k]

        for layer_k, (enc_layer_k, dec_layer_k) in layer_keys_dict.items():
            enc_layers = layers_range_dict[enc_layer_k]
            dec_layers = layers_range_dict[dec_layer_k]

            _score = ctu.trace_attention_manip_uskg_multi_token(
                model=mt.model,
                inp=inp,
                answers_t=answers_t,
                mix_mask_per_layer=dict(
                    [(ctu.layername_uskg(mt.model, 'encoder', l, 'self_attn'), enc_mix_mask) for l in enc_layers] + \
                    [(ctu.layername_uskg(mt.model, 'decoder', l, 'cross_attn'), dec_mix_mask) for l in dec_layers]
                ),
                attn_corrupt_type=attn_corrupt_type,
            ).item()
            result['trace_scores'][sect_k][layer_k] = _score

    return result




def main_sdra_5_0_dirty_attention_vector_effect(args):
    """
    Exp 5.0
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=5_{args.ds}_{args.subject_type}_{args.part}-attn={args.attn_type}-corrupt={args.corrupt_type}-tmp'
    result_save_dir = os.path.join(args.result_dir, 'exp5_0_dirty_attention_vector_effect')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 50
    stride = 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_analysis_sample_dicts(
            mt_uskg, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        input_too_long = False
        if len(analysis_samples) > 0:
            # enc_sentence = analysis_samples[0]['enc_sentence']
            # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
            enc_tokenized = analysis_samples[0]['enc_tokenized']
            input_len = len(enc_tokenized['input_ids'])
            
            if input_len > 500:
                # ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = ctu.add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp5_0_dirty_attention_vector_effect(
                mt_uskg,
                a_ex,
                part=args.part,
                attn_type=args.attn_type,
                corrupt_type=args.corrupt_type,
            )

            ex_results.append(result)

            total_samples += 1
            if result['correct_prediction']:
                n_good_samples += 1

        # ex_out_dict = {
        #     'ex_id': ex_id,
        #     'trace_results': ex_results,
        # }
        ex_out_dict['trace_results'] = ex_results
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()



def main_sdra_5_2_attention_section_removal_effect(args):
    """
    Exp 5.2
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=5.2.1+structcontext_{args.ds}_{args.subject_type}_{args.part}-attn={args.attn_type}-corrupt={args.corrupt_type}'
    result_save_dir = os.path.join(args.result_dir, 'exp5_2_attention_section_removal_effect')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0
    n_too_easy = 0
    n_too_hard = 0

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 20
    stride = 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_analysis_sample_dicts(
            mt_uskg, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        input_too_long = False
        if len(analysis_samples) > 0:
            # enc_sentence = analysis_samples[0]['enc_sentence']
            # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
            enc_tokenized = analysis_samples[0]['enc_tokenized']
            input_len = len(enc_tokenized['input_ids'])
            
            if input_len > 500:
                # ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = ctu.add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp5_2_attention_section_removal_effect(
                mt_uskg,
                a_ex,
                part=args.part,
                attn_type=args.attn_type,
                corrupt_type=args.corrupt_type,
            )

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            elif result['correct_prediction']:
                n_too_easy += 1
            else:
                n_too_hard += 1

        ex_out_dict['trace_results'] = ex_results
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print('total_samples:', total_samples)
    print('n_good_samples:', n_good_samples)
    print('n_too_hard:', n_too_hard)
    print('n_too_easy:', n_too_easy)



def main_sdra_5_3_attention_section_mutual_removal(args):
    """
    Exp 5.3.1
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=5.3.1_{args.ds}_{args.subject_type}_{args.part}-attn={args.attn_type}-corrupt={args.corrupt_type}-tmp'
    result_save_dir = os.path.join(args.result_dir, 'exp5_3_attention_section_mutual_removal')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0
    n_too_easy = 0
    n_too_hard = 0

    f = open(result_save_path, 'w')
    start_id = 0
    # end_id = n_ex
    end_id = 10
    stride = 1
    # stride = 111
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_analysis_sample_dicts(
            mt_uskg, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        input_too_long = False
        if len(analysis_samples) > 0:
            # enc_sentence = analysis_samples[0]['enc_sentence']
            # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
            enc_tokenized = analysis_samples[0]['enc_tokenized']
            input_len = len(enc_tokenized['input_ids'])
            
            if input_len > 500:
                # ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = ctu.add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp5_3_attention_section_mutual_removal(
                mt_uskg,
                a_ex,
                part=args.part,
                attn_type=args.attn_type,
                corrupt_type=args.corrupt_type,
            )

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            elif result['correct_prediction']:
                n_too_easy += 1
            else:
                n_too_hard += 1

        ex_out_dict['trace_results'] = ex_results
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print('total_samples:', total_samples)
    print('n_good_samples:', n_good_samples)
    print('n_too_hard:', n_too_hard)
    print('n_too_easy:', n_too_easy)



def main_sdra_5_4_decoder_cross_attention_removal(args):
    """
    Exp 5.4
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=5.4_{args.ds}_{args.subject_type}-corrupt={args.corrupt_type}'
    result_save_dir = os.path.join(args.result_dir, 'exp5_4_decoder_cross_attention_removal')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0
    n_too_easy = 0
    n_too_hard = 0

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 10
    stride = 1
    # stride = 111
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_analysis_sample_dicts(
            mt_uskg, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        input_too_long = False
        if len(analysis_samples) > 0:
            # enc_sentence = analysis_samples[0]['enc_sentence']
            # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
            enc_tokenized = analysis_samples[0]['enc_tokenized']
            input_len = len(enc_tokenized['input_ids'])
            
            if input_len > 500:
                # ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = ctu.add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp5_4_decoder_cross_attention_removal(
                mt_uskg,
                a_ex,
                # part=args.part,
                # attn_type=args.attn_type,
                corrupt_type=args.corrupt_type,
            )

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            elif result['correct_prediction']:
                n_too_easy += 1
            else:
                n_too_hard += 1

        ex_out_dict['trace_results'] = ex_results
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print('total_samples:', total_samples)
    print('n_good_samples:', n_good_samples)
    print('n_too_hard:', n_too_hard)
    print('n_too_easy:', n_too_easy)



def main_sdra_5_5_both_part_attention_removal(args):
    """
    Exp 5.5 / 5.5.1
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    args.attn_corrupt_type = EXP_5_CONFIGS[args.exp_id]['attn_corrupt_type']

    exp_name = f'exp={args.exp_id}_{args.ds}_{args.subject_type}-attn_crpt={args.attn_corrupt_type}'
    if args.is_tmp:
        exp_name += '-tmp'

    result_save_dir = os.path.join(args.result_dir, 'exp5_5_both_part_attention_removal')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0
    n_too_easy = 0
    n_too_hard = 0

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 10
    stride = 111 if args.is_tmp else 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_analysis_sample_dicts(
            mt_uskg, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        input_too_long = False
        if len(analysis_samples) > 0:
            # enc_sentence = analysis_samples[0]['enc_sentence']
            # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
            enc_tokenized = analysis_samples[0]['enc_tokenized']
            input_len = len(enc_tokenized['input_ids'])
            
            if input_len > 500:
                # ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = ctu.add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp5_5(
                mt_uskg,
                a_ex,
                # part=args.part,
                # attn_type=args.attn_type,
                corrupt_type=args.corrupt_type,
                attn_corrupt_type=args.attn_corrupt_type,
            )

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            elif result['correct_prediction']:
                n_too_easy += 1
            else:
                n_too_hard += 1

        ex_out_dict['trace_results'] = ex_results
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print('total_samples:', total_samples)
    print('n_good_samples:', n_good_samples)
    print('n_too_hard:', n_too_hard)
    print('n_too_easy:', n_too_easy)



def main_sdra_5_attention_corruption(args):
    """
    Exp 5.* unified entry point
    """
    # spider_dataset_path = args.spider_dataset_path
    # spider_db_dir = args.spider_db_dir
    # data_cache_dir = args.data_cache_dir

    exp_config = EXP_5_CONFIGS[args.exp_id]

    args.attn_corrupt_type = exp_config['attn_corrupt_type']
    args.trace_exp_func_id = exp_config['trace_exp_func_id']

    exp_name = f'exp={args.exp_id}_{args.ds}_{args.subject_type}-attn_crpt={args.attn_corrupt_type}'
    if args.is_tmp:
        exp_name += '-tmp'

    result_save_dir = os.path.join(args.result_dir, exp_config['result_save_dir_name'])
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0
    n_too_easy = 0
    n_too_hard = 0

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 10
    stride = 111 if args.is_tmp else 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_analysis_sample_dicts(
            mt_uskg, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        input_too_long = False
        if len(analysis_samples) > 0:
            # enc_sentence = analysis_samples[0]['enc_sentence']
            # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
            enc_tokenized = analysis_samples[0]['enc_tokenized']
            input_len = len(enc_tokenized['input_ids'])
            
            if input_len > 500:
                # ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = ctu.add_clean_prediction(mt_uskg, a_ex)
            
            trace_exp_func = f_reg.FUNC_REGISTRAR['trace_exp'][args.trace_exp_func_id]
            result = trace_exp_func(
                mt_uskg,
                a_ex,
                # part=args.part,
                # attn_type=args.attn_type,
                corrupt_type=args.corrupt_type,
                attn_corrupt_type=args.attn_corrupt_type,
            )

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            elif result['correct_prediction']:
                n_too_easy += 1
            else:
                n_too_hard += 1

        ex_out_dict['trace_results'] = ex_results
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print('total_samples:', total_samples)
    print('n_good_samples:', n_good_samples)
    print('n_too_hard:', n_too_hard)
    print('n_too_easy:', n_too_easy)




EXP_5_CONFIGS = {
    ## Exp 5.3
    '5.3': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp5_3_attention_section_mutual_removal',
        'trace_exp_func_id': '5.3',
    },
    '5.3.1': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp5_3_attention_section_mutual_removal',
        'trace_exp_func_id': '5.3',
    },
    '5.3.2': {
        'attn_corrupt_type': 'logits',  # with re-normalization
        'result_save_dir_name': 'exp5_3_attention_section_mutual_removal',
        'trace_exp_func_id': '5.3',
    },
    ## Exp 5.4
    '5.4': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp5_4_decoder_cross_attention_removal',
        'trace_exp_func_id': '5.4',
    },
    '5.4.1': {
        'attn_corrupt_type': 'logits',  # with re-normalization
        'result_save_dir_name': 'exp5_4_decoder_cross_attention_removal',
        'trace_exp_func_id': '5.4',
    },
    ## Exp 5.5
    '5.5': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp5_5_both_part_attention_removal',
        'trace_exp_func_id': '5.5',
    },
    '5.5.1': {
        'attn_corrupt_type': 'logits',  # with re-normalization
        'result_save_dir_name': 'exp5_5_both_part_attention_removal',
        'trace_exp_func_id': '5.5',
    },
}



def main():
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add different types of arguments
    parser.add_argument('-e', '--exp_id', required=True, help='Experiment ID')
    parser.add_argument('-t', '--is_tmp', action='store_true', help='Do a temp debug run')
    # parser.add_argument('-d', '--arg4', choices=['option1', 'option2', 'option3'], help='An argument with limited choices')
    # parser.add_argument('-e', '--arg5', nargs='+', help='An argument with multiple values')

    # Parse the command-line arguments
    args = parser.parse_args()

    args.ds = 'dev'                     # train, dev
    # args.subject_type = 'column'         # table, table_alias, column, (value)
    # args.part = 'encoder'               # encoder, decoder, both
    # args.attn_type = 'self_attn'
    args.corrupt_type = 'zero'
    args.spider_dataset_path = f'/home/yshao/Projects/SDR-analysis/data/spider/{args.ds}+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'
    args.spider_tables_path = '/home/yshao/Projects/language/language/xsp/data/spider/tables.json'

    args.result_dir = '/home/yshao/Projects/rome/results'

    ctu.evaluate_hardness.evaluator = ctu.load_evaluator(args)

    args.subject_type = 'column'
    main_sdra_5_attention_corruption(args)

    args.subject_type = 'table'
    main_sdra_5_attention_corruption(args)

    args.subject_type = 'table_alias'
    main_sdra_5_attention_corruption(args)



if __name__ == "__main__":
    main()
