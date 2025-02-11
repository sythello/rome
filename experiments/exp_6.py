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


def trace_exp6_0_encoding_corruption_effect_syntax(
    mt,
    a_ex,
):
    """
    Exp6.0: encoding-corruption-effect-syntax
    Corrupting input embedding / final encoding of different input sections - on syntax token predictions (now also include literals)
    Sections: text, struct, cols, tables
    Purpose: check the necessity of each section for syntax prediction. Also, prepare for operational items.
    """

    # expect_input_ranges = a_ex['expect_input_ranges']
    # tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']

    # self_ranges = a_ex['self_ranges']
    # struct_context_ranges = a_ex['context_ranges']
    # self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    # struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]
    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))
    col_name_ranges = a_ex['token_ranges_dict']['col_name_ranges']  # Dict[str, List[(s, e)]]
    col_tok_indices = [i for ranges in col_name_ranges.values() for s, e in ranges for i in range(s, e)]
    table_name_ranges = a_ex['token_ranges_dict']['table_name_ranges']
    table_tok_indices = [i for ranges in table_name_ranges.values() for s, e in ranges for i in range(s, e)]

    result = ctu.make_basic_result_dict(a_ex)

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = result['base_score']
    is_correct_pred = result['correct_prediction']

    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * 11,
        [dec_prompt] * 11,
        answer=expect)

    section_tok_indices_dict = {
        'text': text_tok_indices,
        'struct': struct_tok_indices,
        'columns': col_tok_indices,
        'tables': table_tok_indices,
        'all': text_tok_indices + struct_tok_indices
    }

    # Dict[sect, Dict[layer, states]]
    corrupt_states_dict = {
        sect: {
            'embed': [(tnum, ctu.layername_uskg(mt.model, 'encoder', 0, 'embed')) for tnum in sect_indices],
            'final_enc': [(tnum, ctu.layername_uskg(mt.model, 'encoder', mt.num_enc_layers - 1)) for tnum in sect_indices],
        }
        for sect, sect_indices in section_tok_indices_dict.items()
    }

    # encoder_all_embed_states = [
    #     (tnum, ctu.layername_uskg(mt.model, 'encoder', 0, 'embed'))
    #     for tnum in text_tok_indices + struct_tok_indices
    # ]

    ## Low score: section = all, layer = embed OR final_enc
    # corrupted_vocab_probs: (answer_len, vocab_size)
    corrupted_vocab_probs = ctu.run_repatch_uskg_multi_token(
        model=mt.model,
        inp=inp,
        answer_len=len(answers_t),
        states_to_patch=[],
        states_to_unpatch=[],
        states_to_corrupt=corrupt_states_dict['all']['embed'],
        # answers_t=answers_t,
        replace=True,
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

    # # Just for debugging - cross check
    # low_score_2 = ctu.trace_with_repatch_uskg(
    #     model=mt.model,
    #     inp=inp,
    #     states_to_patch=[],
    #     states_to_unpatch=[],
    #     states_to_corrupt=encoder_all_last_layer_states,
    #     answers_t=answers_t,
    #     replace=True,
    # ).item()
    # assert np.isclose(low_score, low_score_2, atol=1e-4), (low_score, low_score_2)

    # TODO: use "wrong corrupted_answer" instead of 0.5?
    if base_score - low_score < 0.5:
        # n_too_easy += 1
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True

    """ Start exp6.0 scores """

    result['trace_scores'] = {
        'text': dict(),     # 'embed', 'final_enc'; same below
        'struct': dict(),
        'columns': dict(),
        'tables': dict(),
        'all': dict(),
    }

    for sect, sect_d in corrupt_states_dict.items():
        for layer_k, states_to_corrupt in sect_d.items():
            result['trace_scores'][sect][layer_k] = ctu.trace_with_repatch_uskg(
                model=mt.model,
                inp=inp,
                states_to_patch=[],
                states_to_unpatch=[],
                states_to_corrupt=states_to_corrupt,
                answers_t=answers_t,
                replace=True,
            ).item()

    return result


@f_reg.register('trace_exp', '6.1')
def trace_exp6_1_attention_corruption_effect_syntax(
    mt,
    a_ex,                   # output from ctu.create_analysis_sample_dicts()
    part='encoder',         # 'encoder', 'decoder'
    attn_type='self_attn',  # 'self_attn', 'cross_attn'
    corrupt_type='zero',    # 'zero', 'replace', 'add'
    attn_corrupt_type='weights',    # 'weights', 'logits'
    # window=10,              # the window size to try to corrupt
    device='cuda',
):
    """
    AAA
    Exp6.1 (attention-corruption-effect-syntax)
    Check the effect of disabling attention between certain sections (prefix, text, struct), for syntax
    """

    assert (part == 'encoder' and attn_type == 'self_attn'), 'Only Eecoder self-attn implemented'

    # expect_input_ranges = a_ex['expect_input_ranges']
    # tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']

    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))
    prefix_len = mt.model.preseqlen

    result = ctu.make_basic_result_dict(a_ex)

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

    """ Low score of exp 6.1: corrupting all attention weights (same as exp5.3) """

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
    
    att_mix_mask_dict = ctu.build_enc_self_attention_mask(
        a_ex=a_ex,
        seq_len=seq_len,
        prefix_len=prefix_len,
        use_self_node=False
    )

    # att_mix_mask_dict = dict()
    # # mix_mask: (batch, head, src_len, tgt_len)
    
    # t2s_mask = torch.zeros(1, 1, seq_len, seq_len + prefix_len).bool()
    # t2s_mask[:, :, text_st : text_ed, struct_st + prefix_len : struct_ed + prefix_len] = True
    # att_mix_mask_dict['t->s'] = t2s_mask

    # s2t_mask = torch.zeros_like(t2s_mask).bool()
    # s2t_mask[:, :, struct_st : struct_ed, text_st + prefix_len : text_ed + prefix_len] = True
    # att_mix_mask_dict['s->t'] = s2t_mask

    # att_mix_mask_dict['t<->s'] = t2s_mask | s2t_mask

    # t2p_mask = torch.zeros_like(t2s_mask).bool()
    # t2p_mask[:, :, text_st : text_ed, :prefix_len] = True
    # att_mix_mask_dict['t->p'] = t2p_mask

    # s2p_mask = torch.zeros_like(t2s_mask).bool()
    # s2p_mask[:, :, struct_st : struct_ed, :prefix_len] = True
    # att_mix_mask_dict['s->p'] = s2p_mask

    # att_mix_mask_dict['ts->p'] = t2p_mask | s2p_mask

    # # ADDED: section self-attention
    # t2t_mask = torch.zeros_like(t2s_mask).bool()
    # t2t_mask[:, :, text_st : text_ed, text_st + prefix_len : text_ed + prefix_len] = True
    # att_mix_mask_dict['t->t'] = t2t_mask

    # s2s_mask = torch.zeros_like(t2s_mask).bool()
    # s2s_mask[:, :, struct_st : struct_ed, struct_st + prefix_len : struct_ed + prefix_len] = True
    # att_mix_mask_dict['s->s'] = s2s_mask

    # # att_mix_mask_dict['all'] = att_mix_mask_dict['t<->s'] | att_mix_mask_dict['ts->p']
    # att_mix_mask_dict['all'] = torch.ones_like(t2s_mask).bool()

    ## Low score: section = all, layer = embed
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
    
    """ Starting Exp6.1 """

    result['trace_scores'] = dict()
    for mix_k in att_mix_mask_dict.keys():
        result['trace_scores'][mix_k] = {
            # 'window': dict(),             # this & neighbor (-5 to +5) layers attention vector corrupt
            # 'first_layer': None,          # prev exp show that single layer is not so effective
            # 'last_layer': None,
            'low_layers': None,     # 0 - 11
            'high_layers': None,    # 12 - 23
            'all_layers': None,
        }

    for mix_k, mix_mask in att_mix_mask_dict.items():
        # TEMP: only run for newly added sections
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

        # all layers 
        _score = ctu.trace_attention_manip_uskg_multi_token(
            model=mt.model,
            inp=inp,
            answers_t=answers_t,
            mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers)},
            attn_corrupt_type=attn_corrupt_type,
        ).item()
        result['trace_scores'][mix_k]['all_layers'] = _score

    return result


@f_reg.register('trace_exp', '6.2')
def trace_exp6_2_decoder_cross_attention_corruption_syntax(
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
    Exp6.2 (decoder-cross-attention-corruption-syntax)
    (Following exp5.4) Check the effect of disabling decoder cross attention, for syntax
    """

    # assert (part == 'encoder' and attn_type == 'self_attn'), 'Only Eecoder self-attn implemented'
    part = 'decoder'
    attn_type = 'cross_attn'

    # expect_input_ranges = a_ex['expect_input_ranges']
    # tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']

    # text_tok_indices = list(range(*text_range))
    # struct_tok_indices = list(range(*struct_range))
    prefix_len = mt.model.preseqlen

    result = ctu.make_basic_result_dict(a_ex)

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

    """ Low score of exp 6.2: corrupting all attention weights """

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
        use_self_node=False
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
    
    """ Starting Exp6.2 """

    layers_range_dict = {
        'q1_layers': range(0, N_layers // 4),
        'q2_layers': range(N_layers // 4, N_layers // 2),
        'q3_layers': range(N_layers // 2, N_layers * 3 // 4),
        'q4_layers': range(N_layers * 3 // 4, N_layers),
        'low_layers': range(0, N_layers // 2),
        'mid_layers': range(N_layers // 4, N_layers * 3 // 4),
        'high_layers': range(N_layers // 2, N_layers),
        'all_layers': range(N_layers),
    }

    result['trace_scores'] = dict()
    for mix_k in att_mix_mask_dict.keys():
        result['trace_scores'][mix_k] = dict()

    for mix_k, mix_mask in att_mix_mask_dict.items():
        # TEMP: only run for newly added sections
        # if not mix_k.endswith('o'):
        #     continue
        # END TEMP

        for layers_k, layers_range in layers_range_dict.items():
            _score = ctu.trace_attention_manip_uskg_multi_token(
                model=mt.model,
                inp=inp,
                answers_t=answers_t,
                mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in layers_range},
                attn_corrupt_type=attn_corrupt_type,
            ).item()
            result['trace_scores'][mix_k][layers_k] = _score

        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers // 4, N_layers * 3 // 4)},
        # ).item()
        # result['trace_scores'][mix_k]['mid_layers'] = _score

        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers // 2, N_layers)},
        # ).item()
        # result['trace_scores'][mix_k]['high_layers'] = _score

        # # all layers 
        # _score = ctu.trace_attention_manip_uskg_multi_token(
        #     model=mt.model,
        #     inp=inp,
        #     answers_t=answers_t,
        #     mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : mix_mask for l in range(N_layers)},
        # ).item()
        # result['trace_scores'][mix_k]['all_layers'] = _score

    return result



@f_reg.register('trace_exp', '6.3')
def trace_exp6_3(
    mt,
    a_ex,                           # output from ctu.create_analysis_sample_dicts()
    corrupt_type='zero',            # 'zero', 'replace', 'add'
    attn_corrupt_type='weights',    # 'weights', 'logits'
    device='cuda',
):
    """
    AAA
    Exp6.3 (both-part-attention-removal-syntax)
    Check the effect of disabling attention in both encoder and decoder to certain input sections (only text for now)
    """

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

    """ Low score of exp 6.3: corrupting all attention weights """

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
        use_self_node=False
    )

    dec_att_mix_mask_dict = ctu.build_dec_cross_attention_mask(
        a_ex=a_ex,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        prefix_len=prefix_len,
        use_self_node=False
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
    
    """ Starting Exp6.3 """

    # sect_k -> (enc_sect_k, dec_sect_k)
    section_keys_dict = {
        's->t&all->t': ('s->t', 'all->t'),
        # 't->t&all->t': ('t->t', 'all->t'),    # seems trivially need both...
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




def main_sdra_6_0_encoding_corruption_effect_syntax(args):
    """
    Exp 6.0: corrupting encoding (emb / final encoding), check effect on syntax prediction (not including literals)
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=6.0_{args.ds}'
    result_save_dir = os.path.join(args.result_dir, 'exp6_0_encoding_corruption_effect_syntax')
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

        analysis_samples = ctu.create_syntax_analysis_sample_dicts(mt_uskg, ex)

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
            
            result = trace_exp6_0_encoding_corruption_effect_syntax(
                mt_uskg,
                a_ex,
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



def main_sdra_6_1_attention_corruption_effect_syntax(args):
    """
    Exp 6.1: corrupting attention, check effect on syntax prediction (not including literals)
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=6.1_{args.ds}_{args.part}_corrupt={args.corrupt_type}-tmp'
    result_save_dir = os.path.join(args.result_dir, 'exp6_1_attention_corruption_effect_syntax')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ctu.ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0

    f = open(result_save_path, 'w')
    start_id = 0
    # end_id = n_ex
    end_id = 10
    stride = 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_syntax_analysis_sample_dicts(mt_uskg, ex)

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
            
            result = trace_exp6_1_attention_corruption_effect_syntax(
                mt_uskg,
                a_ex,
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



def main_sdra_6_2_decoder_cross_attention_corruption_syntax(args):
    """
    Exp 6.2: corrupting decoder cross attention, check effect on syntax prediction (not including literals)
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=6.2+o_{args.ds}_corrupt={args.corrupt_type}'
    result_save_dir = os.path.join(args.result_dir, 'exp6_2_decoder_cross_attention_corruption_syntax')
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
    # end_id = 10
    stride = 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_syntax_analysis_sample_dicts(mt_uskg, ex)

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
            
            result = trace_exp6_2_decoder_cross_attention_corruption_syntax(
                mt_uskg,
                a_ex,
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



def main_sdra_6_attention_corruption_syntax(args):
    """
    Exp 6.*
    """
    # spider_dataset_path = args.spider_dataset_path
    # spider_db_dir = args.spider_db_dir
    # data_cache_dir = args.data_cache_dir

    exp_config = EXP_6_CONFIGS[args.exp_id]

    args.attn_corrupt_type = exp_config['attn_corrupt_type']
    args.trace_exp_func_id = exp_config['trace_exp_func_id']

    exp_name = f'exp={args.exp_id}_{args.ds}-attn_crpt={args.attn_corrupt_type}'
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

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 10
    stride = 111 if args.is_tmp else 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = ctu.create_syntax_analysis_sample_dicts(mt_uskg, ex)

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
                attn_corrupt_type=args.attn_corrupt_type,
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




EXP_6_CONFIGS = {
    ## Exp 6.1
    '6.1': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp6_1_attention_corruption_effect_syntax',
        'trace_exp_func_id': '6.1',
    },
    '6.1.1': {
        'attn_corrupt_type': 'logits',  # with re-normalization
        'result_save_dir_name': 'exp6_1_attention_corruption_effect_syntax',
        'trace_exp_func_id': '6.1',
    },
    ## Exp 6.2
    '6.2': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp6_2_decoder_cross_attention_corruption_syntax',
        'trace_exp_func_id': '6.2',
    },
    '6.2.1': {
        'attn_corrupt_type': 'logits',  # with re-normalization
        'result_save_dir_name': 'exp6_2_decoder_cross_attention_corruption_syntax',
        'trace_exp_func_id': '6.2',
    },
    ## Exp 6.3
    '6.3': {
        'attn_corrupt_type': 'weights',  # no re-normalization
        'result_save_dir_name': 'exp6_3_both_part_attention_corruption_syntax',
        'trace_exp_func_id': '6.3',
    },
    '6.3.1': {
        'attn_corrupt_type': 'logits',  # with re-normalization
        'result_save_dir_name': 'exp6_3_both_part_attention_corruption_syntax',
        'trace_exp_func_id': '6.3',
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

    main_sdra_6_attention_corruption_syntax(args)



if __name__ == "__main__":
    main()
