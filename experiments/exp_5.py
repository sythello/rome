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



def trace_exp5_3_attention_section_mutual_removal(
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
    Exp5.3.1 (attention-section-mutual-removal)
    Check the effect of disabling attention between certain sections (prefix, text, struct)
    """

    assert (part == 'encoder' and attn_type == 'self_attn'), 'Only Eecoder self-attn implemented'

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

    text_st, text_ed = text_range
    struct_st, struct_ed = struct_range
    
    att_mix_mask_dict = dict()
    # mix_mask: (batch, head, src_len, tgt_len)
    
    t2s_mask = torch.zeros(1, 1, seq_len, seq_len + prefix_len).bool()
    t2s_mask[:, :, text_st : text_ed, struct_st + prefix_len : struct_ed + prefix_len] = True
    att_mix_mask_dict['t->s'] = t2s_mask

    s2t_mask = torch.zeros_like(t2s_mask).bool()
    s2t_mask[:, :, struct_st : struct_ed, text_st + prefix_len : text_ed + prefix_len] = True
    att_mix_mask_dict['s->t'] = s2t_mask

    att_mix_mask_dict['t<->s'] = t2s_mask | s2t_mask

    t2p_mask = torch.zeros_like(t2s_mask).bool()
    t2p_mask[:, :, text_st : text_ed, :prefix_len] = True
    att_mix_mask_dict['t->p'] = t2p_mask

    s2p_mask = torch.zeros_like(t2s_mask).bool()
    s2p_mask[:, :, struct_st : struct_ed, :prefix_len] = True
    att_mix_mask_dict['s->p'] = s2p_mask

    att_mix_mask_dict['ts->p'] = t2p_mask | s2p_mask

    # ADDED: section self-attention
    t2t_mask = torch.zeros_like(t2s_mask).bool()
    t2t_mask[:, :, text_st : text_ed, text_st + prefix_len : text_ed + prefix_len] = True
    att_mix_mask_dict['t->t'] = t2t_mask

    s2s_mask = torch.zeros_like(t2s_mask).bool()
    s2s_mask[:, :, struct_st : struct_ed, struct_st + prefix_len : struct_ed + prefix_len] = True
    att_mix_mask_dict['s->s'] = s2s_mask

    # ADDED: regarding struct context
    # Notice that it's ok to have 1 list in indexing, but not ok to have 2
    # If there are 2 lists, it will become a "gather()" which treats the 2 lists in a zipped way
    s2c_mask = torch.zeros_like(t2s_mask).bool()
    s2c_mask[:, :, struct_st : struct_ed, struct_context_tok_indices_tgt_side] = True
    att_mix_mask_dict['s->c'] = s2c_mask

    c2p_mask = torch.zeros_like(t2s_mask).bool()
    c2p_mask[:, :, struct_context_tok_indices, :prefix_len] = True
    att_mix_mask_dict['c->p'] = c2p_mask

    # c2t: skipped, as already see even s2t is not so effective

    c2s_mask = torch.zeros_like(t2s_mask).bool()
    c2s_mask[:, :, struct_context_tok_indices, struct_st + prefix_len : struct_ed + prefix_len] = True
    att_mix_mask_dict['c->s'] = c2s_mask

    c2c_mask = c2s_mask.clone()
    c2c_mask[:, :, :, self_tok_indices_tgt_side] = False
    assert c2c_mask.sum().item() == len(struct_context_tok_indices) ** 2, \
        (c2c_mask.sum().item(), len(struct_context_tok_indices) ** 2)
    att_mix_mask_dict['c->c'] = c2c_mask

    # att_mix_mask_dict['all'] = att_mix_mask_dict['t<->s'] | att_mix_mask_dict['ts->p']
    att_mix_mask_dict['all'] = torch.ones_like(t2s_mask).bool()

    # breakpoint()

    low_score = result['low_score'] = ctu.trace_attention_manip_uskg_multi_token(
        model=mt.model,
        inp=inp,
        answers_t=answers_t,
        mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : att_mix_mask_dict['all'] for l in range(N_layers)},
        # layers_to_mix=[ctu.layername_uskg(mt.model, part, l, attn_type) for l in range(N_layers)],
        # src_tokens_to_mix=self_tok_indices,
        # tgt_tokens_to_mix=att_tgt_sections_dict['all'],
        # noise=noise,
        # replace=replace,
    ).item()

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp5.3 """

    result['trace_scores'] = dict()
    for mix_k in att_mix_mask_dict.keys():
        result['trace_scores'][mix_k] = {
            'window': dict(),           # this & neighbor (-5 to +5) layers attention vector corrupt
            'first_layer': None,
            'last_layer': None,
            'all_layers': None,
        }

    for mix_k, mix_mask in att_mix_mask_dict.items():
        # TEMP: only run for newly added sections
        # if 'c' not in mix_k:
        if mix_k != 'c->p':
            continue
        # END TEMP

        # window (for speed, only compute on a subset of layers)
        # for layer_id in range(N_layers):
        for layer_id in range(3, N_layers, 4):
            _score = ctu.trace_attention_manip_uskg_multi_token(
                model=mt.model,
                inp=inp,
                answers_t=answers_t,
                mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask
                               for l in range(
                                    max(0, layer_id - window // 2),
                                    min(N_layers, layer_id + window // 2)
                                )},
            ).item()
            result['trace_scores'][mix_k]['window'][layer_id] = _score
        
        # single layer 
        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, 0, attn_type) : mix_mask},
        ).item()
        result['trace_scores'][mix_k]['first_layer'] = _score

        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, N_layers - 1, attn_type) : mix_mask},
        ).item()
        result['trace_scores'][mix_k]['last_layer'] = _score

        # all layers 
        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers)},
        ).item()
        result['trace_scores'][mix_k]['all_layers'] = _score

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

    exp_name = f'exp=5.3.1+c2p_{args.ds}_{args.subject_type}_{args.part}-attn={args.attn_type}-corrupt={args.corrupt_type}'
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
    end_id = n_ex
    # end_id = 20
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



def main():
    args = Namespace()
    args.ds = 'dev'                     # train, dev
    # args.subject_type = 'column'         # table, table_alias, column, (value)
    args.part = 'encoder'               # encoder, decoder, both
    args.attn_type = 'self_attn'
    args.corrupt_type = 'zero'
    args.spider_dataset_path = f'/home/yshao/Projects/SDR-analysis/data/spider/{args.ds}+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'
    args.spider_tables_path = '/home/yshao/Projects/language/language/xsp/data/spider/tables.json'

    args.result_dir = '/home/yshao/Projects/rome/results'

    ctu.evaluate_hardness.evaluator = ctu.load_evaluator(args)

    args.subject_type = 'column'
    main_sdra_5_3_attention_section_mutual_removal(args)

    args.subject_type = 'table'
    main_sdra_5_3_attention_section_mutual_removal(args)

    args.subject_type = 'table_alias'
    main_sdra_5_3_attention_section_mutual_removal(args)



if __name__ == "__main__":
    main()
