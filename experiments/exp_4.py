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


def trace_attention_uskg(
    model,  # The model
    inp,    # A batch of input (can just have bsz=1)
    answer_len,
    trace_layers=None,  # List of traced outputs to return
):
    """ Can work for multi-token prediction. Encoder results are the same; decoder is causal LM, therefore the intermediate results are valid even when there are following tokens. """

    with torch.no_grad(), nethook.TraceDict(
        model,
        layers=trace_layers or [],
        edit_output=None,
    ) as td:
        outputs_exp = ctu.run_model_forward_uskg(model, **inp, output_attentions=True)

    # already checked that prefix comes before the actual sequence
    # layers, (bsz, n_heads, seq_len, prev_len + seq_len)
    encoder_attentions = outputs_exp.encoder_attentions

    # layers, (bsz, n_heads, prompt_len, prev_len + seq_len)
    cross_attentions = outputs_exp.cross_attentions

    # layers, (bsz, n_heads, prompt_len, prev_len + prompt_len)
    decoder_attentions = outputs_exp.decoder_attentions

    ans_probs = torch.softmax(outputs_exp.logits[:, -answer_len:, :], dim=-1)

    # traced_outputs = {l : td[l].output for l in trace_layers}
    attention_dict = {
        'encoder_attentions': encoder_attentions,
        'cross_attentions': cross_attentions,
        'decoder_attentions': decoder_attentions,
    }

    return ans_probs, attention_dict



def trace_exp4_inspect_attention(
    mt,
    a_ex,           # output from ctu.create_analysis_sample_dicts()
    # part='both',    # 'encoder', 'decoder', 'both'
    device='cuda',
):
    """
    AAA
    Exp4 (inspect attention)
    """

    expect_input_ranges = a_ex['expect_input_ranges']
    tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence],
        [dec_prompt],
        answer=expect,
        device=device
    )
    answer_len = 1
    if expect is not None:
        answer_len = len(mt.tokenizer.tokenize(expect))

    # trace_layers = [...]
    
    # ans_probs: (bsz, ans_len, vocab)
    ans_probs, attention_dict = trace_attention_uskg(
        mt.model,
        inp,
        answer_len=answer_len,
    )

    # with torch.no_grad():
    #     answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]

    probs, answers_t = torch.max(ans_probs.squeeze(0), dim=-1)

    answers_t = answers_t.detach().cpu().numpy().tolist()
    probs = probs.detach().cpu().numpy().tolist()
    base_score = min(probs)
    answer = ctu.decode_sentences(mt.tokenizer, answers_t)
    is_correct_pred = (answer.strip() == expect)

    result_dict = ctu.make_basic_result_dict(a_ex)
    result_dict['answer'] = answer
    result_dict['probs'] = probs
    result_dict['base_score'] = base_score
    result_dict['answers_t'] = answers_t
    result_dict['correct_prediction'] = is_correct_pred
    
    # Prune attention dicts (otherwise it's too large)

    # encoder_attentions: layers, (bsz, n_heads, seq_len, prev_len + seq_len)
    encoder_attentions = attention_dict['encoder_attentions']
    encoder_attentions = torch.stack(encoder_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, seq_len, prev_len + seq_len)
    save_enc_attn = encoder_attentions[:, :, tok_indices, :].detach().cpu().numpy().tolist()
    save_enc_attn = ctu.nested_list_processing(save_enc_attn, func=lambda x: np.format_float_positional(x, precision=2))
    save_enc_head_t = inp['input_ids'][0][tok_indices]
    save_enc_head_tokens = ctu.decode_tokens(mt.tokenizer, save_enc_head_t)
    save_enc_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['input_ids'][0])
    
    # cross_attentions: layers, (bsz, n_heads, prompt_len, prev_len + seq_len)
    cross_attentions = attention_dict['cross_attentions']
    cross_attentions = torch.stack(cross_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, prompt_len, prev_len + seq_len)
    save_cross_attn = encoder_attentions[:, :, -answer_len:, :].detach().cpu().numpy().tolist()
    # save_cross_head_t = inp['decoder_input_ids'][0][-answer_len:]  # this is ground truth (should == answers_t if correct)
    save_cross_attn = ctu.nested_list_processing(save_cross_attn, func=lambda x: np.format_float_positional(x, precision=2))
    save_cross_head_tokens = ctu.decode_tokens(mt.tokenizer, answers_t)
    save_cross_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['input_ids'][0])
    
    # decoder_attentions: layers, (bsz, n_heads, prompt_len, prev_len + seq_len)
    decoder_attentions = attention_dict['decoder_attentions']
    decoder_attentions = torch.stack(decoder_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, prompt_len, prev_len + seq_len)
    save_dec_attn = decoder_attentions[:, :, -answer_len:, :].detach().cpu().numpy().tolist()
    save_dec_attn = ctu.nested_list_processing(save_dec_attn, func=lambda x: np.format_float_positional(x, precision=2))
    # save_dec_head_t = inp['decoder_input_ids'][0][-answer_len:]  # this is ground truth (should == answers_t if correct)
    save_dec_head_tokens = ctu.decode_tokens(mt.tokenizer, answers_t)
    save_dec_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['decoder_input_ids'][0])
    
    result_dict['attentions'] = {
        'enc': {
            'attn': save_enc_attn,
            'head_tokens': save_enc_head_tokens,
            'cand_tokens': save_enc_cand_tokens,
        },
        'cross': {
            'attn': save_cross_attn,
            'head_tokens': save_cross_head_tokens,
            'cand_tokens': save_cross_cand_tokens,
        },
        'dec': {
            'attn': save_dec_attn,
            'head_tokens': save_dec_head_tokens,
            'cand_tokens': save_dec_cand_tokens,
        }
    }

    return result_dict


# def plot_uskg_attention():
#     """
#     X: input token
#     Y: layer
#     (code is in notebook)
#     """
    
#     pass

# def trace_exp4_1_dirty_attention_effect(
#     mt,
#     a_ex,                   # analysis_sample (a_ex) with `add_clean_prediction()`
#     samples=10,
#     noise=0.1,
#     part='encoder',
#     kind='self_attn',
#     uniform_noise=False,
#     replace=True,
#     window=10,
# ):
#     """
#     AAA
#     Exp 4.1
#     """

#     text_in = a_ex['text_in']
#     struct_in = a_ex['struct_in']
#     enc_sentence = a_ex['enc_sentence']
#     enc_tokenized = a_ex['enc_tokenized']
#     text_range = a_ex['text_range']
#     struct_range = a_ex['struct_range']

#     expect = a_ex['expect']
#     # dec_prompt = make_dec_prompt(a_ex['seq_out'], expect)
#     dec_prompt = a_ex['dec_prompt']

#     parsed_struct_in = a_ex['parsed_struct_in']
#     col2table = a_ex['col2table']
#     node_name_ranges = a_ex['node_name_ranges']
#     subject_type = a_ex['expect_type']
#     expect_input_ranges = a_ex['expect_input_ranges']

#     # if len(expect_input_ranges) > 1:
#     #     assert not a_ex['remove_struct_duplicate_nodes']
#     #     raise NotImplementedError
    
#     # node_range, = expect_input_ranges

#     if part != 'encoder':
#         raise ValueError(part)
    
#     text_range = a_ex['text_range']
#     struct_range = a_ex['struct_range']

#     # For full context tokens, use [0, L] and [R, -1]
#     # L: node left max end index ; R: node right min start index

#     token_ranges_dict = a_ex['token_ranges_dict']

#     expect_input_ranges = a_ex['expect_input_ranges']    # list of ranges of node-of-interest (code allows dup)
#     enc_sentence = a_ex['enc_sentence']
#     dec_prompt = a_ex['dec_prompt']

#     self_ranges = a_ex['self_ranges']
#     context_ranges = a_ex['context_ranges']

#     self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
#     context_tok_indices = corrupt_tok_indices = [i for s, e in context_ranges for i in range(s, e)]
#     text_tok_indices = list(range(*text_range))


#     result = ctu.make_basic_result_dict(a_ex)
#     result['self_ranges'] = self_ranges
#     result['struct_context_ranges'] = context_ranges

#     result['trace_scores'] = {
#         'single_layer_corrupt': dict(),     # this layer self + text patch, next layer context patch
#         'low_layers_restore': dict(),       # this layer everything patch, next layer context unpatch
#         'high_layers_restore': dict(),      # this & all above layers context patch
#         'single_layer_restore': dict(),     # this layer context patch, next layer context unpatch
#         'temp1': dict(),                    # temp1 (single_layer_restore with no unpatch): this layer context patch
#     }

#     inp = ctu.make_inputs_t5(
#         mt.tokenizer,
#         [enc_sentence] * (1 + samples),
#         [dec_prompt] * (1 + samples),
#         answer=expect)

#     # encoder_struct_no_node_last_layer_states = [
#     #     (tnum, layername_uskg(mt.model, 'encoder', mt.num_enc_layers - 1))
#     #     for tnum in struct_no_node_toks
#     # ]

#     # answer_len = len(mt.tokenizer.tokenize(expect))
#     # answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]
#     # base_score = min(base_score).item()
#     # answer = decode_sentences(mt.tokenizer, answers_t)
    
#     answer = result['answer']
#     answers_t = result['answers_t']
#     base_score = a_ex['base_score']
#     is_correct_pred = result['correct_prediction']
    
#     if not is_correct_pred:
#         # scores don't make sense when clean pred is wrong 
#         result['is_good_sample'] = False
#         return result
    
#     base_score = result['base_score']
#     low_score = result['low_score'] = ctu.trace_with_repatch_uskg(
#         model=mt.model,
#         inp=inp,
#         states_to_patch=[],
#         states_to_unpatch=[],
#         answers_t=answers_t,
#         tokens_to_mix=corrupt_tok_indices,
#         tokens_to_mix_individual_indices=True,
#         replace=True,
#     ).item()

#     # if base_score < 0.5:
#     if answer.strip() != expect:
#         assert False, "Incorrect prediction should already been skipped!"
    
#     ## If base and low score has no diff, "too easy", skip
#     if base_score - low_score < 0.5:
#         result['is_good_sample'] = False
#         return result
    
#     result['is_good_sample'] = True
    

#     """ Starting Exp3.1 """
#     for layer_id in range(mt.num_enc_layers):
#         # self_tok_indices, corrupt_tok_indices, layer_id
#         _curr_layer_self = [(tnum, ctu.layername_uskg(mt.model, 'encoder', layer_id))
#                             for tnum in self_tok_indices]
#         _curr_layer_text = [(tnum, ctu.layername_uskg(mt.model, 'encoder', layer_id))
#                             for tnum in text_tok_indices]
#         _curr_layer_ctx = [(tnum, ctu.layername_uskg(mt.model, 'encoder', layer_id))
#                             for tnum in context_tok_indices]
#         _next_layer_ctx = [(tnum, ctu.layername_uskg(mt.model, 'encoder', layer_id + 1))
#                             for tnum in context_tok_indices] if layer_id < mt.num_enc_layers - 1 else []
#         _above_layers_ctx = [(tnum, ctu.layername_uskg(mt.model, 'encoder', l))
#                              for tnum in context_tok_indices
#                              for l in range(layer_id + 1, mt.num_enc_layers)]

#         # single_layer_corrupt: this layer self + text patch, next layer context patch
#         _score = ctu.trace_with_repatch_uskg(
#             model=mt.model,
#             inp=inp,
#             states_to_patch=_curr_layer_self + _curr_layer_text + _next_layer_ctx,
#             states_to_unpatch=[],
#             answers_t=answers_t,
#             tokens_to_mix=corrupt_tok_indices,
#             tokens_to_mix_individual_indices=True,
#             replace=True,
#         ).item()
#         result['trace_scores']['single_layer_corrupt'][layer_id] = _score

#         # low_layers_restore: this layer self + text patch
#         # (new) this layer everything patch, next layer context unpatch
#         _score = ctu.trace_with_repatch_uskg(
#             model=mt.model,
#             inp=inp,
#             states_to_patch=_curr_layer_self + _curr_layer_text + _curr_layer_ctx,
#             states_to_unpatch=_next_layer_ctx,
#             answers_t=answers_t,
#             tokens_to_mix=corrupt_tok_indices,
#             tokens_to_mix_individual_indices=True,
#             replace=True,
#         ).item()
#         result['trace_scores']['low_layers_restore'][layer_id] = _score

#         # high_layers_restore: this & all above layers context patch
#         _score = ctu.trace_with_repatch_uskg(
#             model=mt.model,
#             inp=inp,
#             states_to_patch=_curr_layer_ctx + _above_layers_ctx,
#             states_to_unpatch=[],
#             answers_t=answers_t,
#             tokens_to_mix=corrupt_tok_indices,
#             tokens_to_mix_individual_indices=True,
#             replace=True,
#         ).item()
#         result['trace_scores']['high_layers_restore'][layer_id] = _score

#         # single_layer_restore: this layer context patch, next layer context unpatch
#         _score = ctu.trace_with_repatch_uskg(
#             model=mt.model,
#             inp=inp,
#             states_to_patch=_curr_layer_ctx,
#             states_to_unpatch=_next_layer_ctx,
#             answers_t=answers_t,
#             tokens_to_mix=corrupt_tok_indices,
#             tokens_to_mix_individual_indices=True,
#             replace=True,
#         ).item()
#         result['trace_scores']['single_layer_restore'][layer_id] = _score

#         # temp1 (single_layer_restore with no unpatch): this layer context patch
#         _score = ctu.trace_with_repatch_uskg(
#             model=mt.model,
#             inp=inp,
#             states_to_patch=_curr_layer_ctx,
#             states_to_unpatch=[],
#             answers_t=answers_t,
#             tokens_to_mix=corrupt_tok_indices,
#             tokens_to_mix_individual_indices=True,
#             replace=True,
#         ).item()
#         result['trace_scores']['temp1'][layer_id] = _score

#     return result


def trace_exp4_1(
    mt,
    a_ex,           # output from ctu.create_analysis_sample_dicts_all_nodes()
    # part='encoder',    # 'encoder', 'decoder', 'both'
    device='cuda',
):
    """
    AAA
    Exp4.1 (inspect attention distribution for all encoder nodes)
    """

    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    token_ranges_dict = a_ex['token_ranges_dict']
    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']
    text_st, text_ed = text_range
    struct_st, struct_ed = struct_range

    preseqlen = mt.model.preseqlen

    # expect_input_ranges = a_ex['expect_input_ranges']
    # tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    # expect = a_ex['expect']     # for funcs to work, not actually used here in 4.1

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence],
        [dec_prompt],
        answer=None, 
        device=device
    )
    answer_len = 1      # for funcs to work, not actually used here in 4.1
    # if expect is not None:
    #     answer_len = len(mt.tokenizer.tokenize(expect))

    # trace_layers = [...]
    
    # ans_probs: (bsz, ans_len, vocab)
    ans_probs, attention_dict = trace_attention_uskg(
        mt.model,
        inp,
        answer_len=answer_len,
    )

    ## No need to (actually cannot) check correctness
    # with torch.no_grad():
    #     answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]
    # probs, answers_t = torch.max(ans_probs.squeeze(0), dim=-1)
    # answers_t = answers_t.detach().cpu().numpy().tolist()
    # probs = probs.detach().cpu().numpy().tolist()
    # base_score = min(probs)
    # answer = ctu.decode_sentences(mt.tokenizer, answers_t)
    # is_correct_pred = (answer.strip() == expect)
    result_dict = ctu.make_basic_result_dict(a_ex)
    # result_dict['answer'] = answer
    # result_dict['probs'] = probs
    # result_dict['base_score'] = base_score
    # result_dict['answers_t'] = answers_t
    # result_dict['correct_prediction'] = is_correct_pred

    add_keys = ['occ_cols', 'non_occ_cols', 'occ_tabs', 'non_occ_tabs']
    for k in add_keys:
        result_dict[k] = a_ex[k]
    
    # Prune attention dicts (otherwise it's too large)

    # encoder_attentions: layers, (bsz, n_heads, seq_len, prev_len + seq_len)
    encoder_attentions = attention_dict['encoder_attentions']
    encoder_attentions = torch.stack(encoder_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, seq_len, prev_len + seq_len)

    all_cols = a_ex['occ_cols'] + a_ex['non_occ_cols']
    all_tabs = a_ex['occ_tabs'] + a_ex['non_occ_tabs']

    result_dict['attentions'] = {
        'col': dict(),
        'tab': dict(),
    }

    for col in all_cols:
        first_tok_pos = token_ranges_dict['col_name_ranges'][col][0][0]

        _sect_att_weights = dict()
        for i in range(preseqlen):
            # size = (layers, n_heads); same below
            _attn = encoder_attentions[:, :, first_tok_pos, i]
            _sect_att_weights[f'prefix#{i}'] = _attn
        
        _text_attn = encoder_attentions[:, :, first_tok_pos, preseqlen + text_st : preseqlen + text_ed].sum(dim=-1)
        _sect_att_weights['text'] = _text_attn

        _self_ranges = a_ex['col_self_ranges'][col]
        _self_indices = [preseqlen + i for s, e in _self_ranges for i in range(s, e)]
        _self_attn = encoder_attentions[:, :, first_tok_pos, _self_indices].sum(dim=-1)
        _sect_att_weights['self'] = _self_attn

        ## context indices doesn't include EOS and mid prompt
        _context_ranges = a_ex['col_context_ranges'][col]
        _context_indices = [preseqlen + i for s, e in _context_ranges for i in range(s, e)]
        _context_attn = encoder_attentions[:, :, first_tok_pos, _context_indices].sum(dim=-1)
        _sect_att_weights['context'] = _context_attn

        ## ADD: "others" as a section (including EOS and mid prompt)
        _mid_attn = encoder_attentions[:, :, first_tok_pos, preseqlen + text_ed : preseqlen + struct_st].sum(dim=-1)
        _eos_attn = encoder_attentions[:, :, first_tok_pos, -1]
        _sect_att_weights['others'] = _mid_attn + _eos_attn

        for k, v in _sect_att_weights.items():
            v = v.detach().cpu().numpy().tolist()
            _sect_att_weights[k] = ctu.nested_list_processing(v, func=lambda x: np.format_float_positional(x, precision=2))

        result_dict['attentions']['col'][col] = _sect_att_weights


    for tab in all_tabs:
        first_tok_pos = token_ranges_dict['table_name_ranges'][tab][0][0]

        _sect_att_weights = dict()
        for i in range(preseqlen):
            # size = (layers, n_heads); same below
            _attn = encoder_attentions[:, :, first_tok_pos, i]
            _sect_att_weights[f'prefix#{i}'] = _attn
        
        _text_attn = encoder_attentions[:, :, first_tok_pos, preseqlen + text_st : preseqlen + text_ed].sum(dim=-1)
        _sect_att_weights['text'] = _text_attn

        _self_ranges = a_ex['tab_self_ranges'][tab]
        _self_indices = [preseqlen + i for s, e in _self_ranges for i in range(s, e)]
        _self_attn = encoder_attentions[:, :, first_tok_pos, _self_indices].sum(dim=-1)
        _sect_att_weights['self'] = _self_attn

        ## context indices doesn't include EOS and mid prompt
        _context_ranges = a_ex['tab_context_ranges'][tab]
        _context_indices = [preseqlen + i for s, e in _context_ranges for i in range(s, e)]
        _context_attn = encoder_attentions[:, :, first_tok_pos, _context_indices].sum(dim=-1)
        _sect_att_weights['context'] = _context_attn

        ## ADD: "others" as a section (including EOS and mid prompt)
        _mid_attn = encoder_attentions[:, :, first_tok_pos, preseqlen + text_ed : preseqlen + struct_st].sum(dim=-1)
        _eos_attn = encoder_attentions[:, :, first_tok_pos, -1]
        _sect_att_weights['others'] = _mid_attn + _eos_attn

        for k, v in _sect_att_weights.items():
            v = v.detach().cpu().numpy().tolist()
            _sect_att_weights[k] = ctu.nested_list_processing(v, func=lambda x: np.format_float_positional(x, precision=2))

        result_dict['attentions']['tab'][tab] = _sect_att_weights

    # save_enc_attn = encoder_attentions[:, :, tok_indices, :].detach().cpu().numpy().tolist()
    # save_enc_attn = ctu.nested_list_processing(save_enc_attn, func=lambda x: np.format_float_positional(x, precision=2))
    # save_enc_head_t = inp['input_ids'][0][tok_indices]
    # save_enc_head_tokens = ctu.decode_tokens(mt.tokenizer, save_enc_head_t)
    # save_enc_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['input_ids'][0])
    

    return result_dict



def main_sdra_4_inspect_attention(args):
    """
    Exp 4: Check the attention weights
    Purpose: directly look for information flow patterns
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=4_{args.ds}_{args.subject_type}_{args.part}-tmp'
    result_save_dir = os.path.join(args.result_dir, 'exp4_inspect_attention')
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
    stride = 10
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

            # a_ex = add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp4_inspect_attention(mt_uskg, a_ex)

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




def main_sdra_4_1_attention_weights_distribution(args):
    """
    Exp 4.1: Check the attention weights distribution for all nodes
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=4.1_{args.ds}'
    if args.is_tmp:
        exp_name += '-tmp'

    result_save_dir = os.path.join(args.result_dir, 'exp4_1_attention_weights_distribution')
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
    stride = 111 if args.is_tmp else 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_ex = ctu.create_analysis_sample_dicts_all_nodes(
            mt_uskg,
            ex,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        result = trace_exp4_1(mt_uskg, analysis_ex)

        ex_out_dict['trace_results'] = result

        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_id', required=True, help='Experiment ID')
    parser.add_argument('-t', '--is_tmp', action='store_true', help='Do a temp debug run')
    
    args = parser.parse_args()

    args.ds = 'dev'                     # train, dev
    # args.subject_type = 'column'         # table, table_alias, column, (value)
    # args.part = 'both'               # encoder, decoder, both
    args.spider_dataset_path = f'/home/yshao/Projects/SDR-analysis/data/spider/{args.ds}+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'
    args.spider_tables_path = '/home/yshao/Projects/language/language/xsp/data/spider/tables.json'

    args.result_dir = '/home/yshao/Projects/rome/results'

    ctu.evaluate_hardness.evaluator = ctu.load_evaluator(args)

    if args.exp_id == '4':
        for subject_type in ['column', 'table', 'table_alias']:
            args.subject_type = subject_type
            main_sdra_4_inspect_attention(args)

    elif args.exp_id == '4.1':
        main_sdra_4_1_attention_weights_distribution(args)

    else:
        raise ValueError(args.exp_id)


if __name__ == "__main__":
    main()
