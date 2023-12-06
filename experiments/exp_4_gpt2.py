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
from experiments import causal_trace_uskg_gpt2 as ctu2

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


def trace_attention_uskg_gpt2(
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
        outputs_exp = ctu2.run_model_forward_uskg_gpt2(model, **inp, output_attentions=True)

    # already checked that prefix comes before the actual sequence
    # GPT2 also has this format: Tuple(layers, Tensor(bsz, n_heads, seq_len, prev_len + seq_len))
    attentions = outputs_exp.attentions
    # encoder_attentions = outputs_exp.encoder_attentions
    # cross_attentions = outputs_exp.cross_attentions
    # decoder_attentions = outputs_exp.decoder_attentions

    ans_probs = torch.softmax(outputs_exp.logits[:, -answer_len:, :], dim=-1)

    # traced_outputs = {l : td[l].output for l in trace_layers}
    attention_dict = {
        'attentions': attentions,
    }

    return ans_probs, attention_dict



def trace_exp4_inspect_attention_gpt2(
    mt,
    a_ex,           # output from ctu.create_analysis_sample_dicts()
    # part='both',    # 'encoder', 'decoder', 'both'
    device='cuda',
):
    """
    AAA
    GPT2 Exp4 (inspect attention)
    """

    expect_input_ranges = a_ex['expect_input_ranges']
    tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    inp = ctu2.make_inputs_gpt2(
        mt.tokenizer,
        # [enc_sentence],
        [dec_prompt],
        answer=expect,
        device=device
    )
    answer_len = 1
    if expect is not None:
        answer_len = len(mt.tokenizer.tokenize(expect))

    # trace_layers = [...]
    
    # ans_probs: (bsz, ans_len, vocab)
    ans_probs, attention_dict = trace_attention_uskg_gpt2(
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

    # # encoder_attentions: layers, (bsz, n_heads, seq_len, prev_len + seq_len)
    # encoder_attentions = attention_dict['encoder_attentions']
    # encoder_attentions = torch.stack(encoder_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, seq_len, prev_len + seq_len)
    # save_enc_attn = encoder_attentions[:, :, tok_indices, :].detach().cpu().numpy().tolist()
    # save_enc_attn = ctu.nested_list_processing(save_enc_attn, func=lambda x: np.format_float_positional(x, precision=2))
    # save_enc_head_t = inp['input_ids'][0][tok_indices]
    # save_enc_head_tokens = ctu.decode_tokens(mt.tokenizer, save_enc_head_t)
    # save_enc_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['input_ids'][0])
    
    # # cross_attentions: layers, (bsz, n_heads, prompt_len, prev_len + seq_len)
    # cross_attentions = attention_dict['cross_attentions']
    # cross_attentions = torch.stack(cross_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, prompt_len, prev_len + seq_len)
    # save_cross_attn = encoder_attentions[:, :, -answer_len:, :].detach().cpu().numpy().tolist()
    # # save_cross_head_t = inp['decoder_input_ids'][0][-answer_len:]  # this is ground truth (should == answers_t if correct)
    # save_cross_attn = ctu.nested_list_processing(save_cross_attn, func=lambda x: np.format_float_positional(x, precision=2))
    # save_cross_head_tokens = ctu.decode_tokens(mt.tokenizer, answers_t)
    # save_cross_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['input_ids'][0])
    
    # # decoder_attentions: layers, (bsz, n_heads, prompt_len, prev_len + seq_len)
    # decoder_attentions = attention_dict['decoder_attentions']
    # decoder_attentions = torch.stack(decoder_attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, prompt_len, prev_len + seq_len)
    # save_dec_attn = decoder_attentions[:, :, -answer_len:, :].detach().cpu().numpy().tolist()
    # save_dec_attn = ctu.nested_list_processing(save_dec_attn, func=lambda x: np.format_float_positional(x, precision=2))
    # # save_dec_head_t = inp['decoder_input_ids'][0][-answer_len:]  # this is ground truth (should == answers_t if correct)
    # save_dec_head_tokens = ctu.decode_tokens(mt.tokenizer, answers_t)
    # save_dec_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['decoder_input_ids'][0])
    
    attentions = attention_dict['attentions']
    attentions = torch.stack(attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, prompt_len, prev_len + seq_len)
    save_attn = attentions[:, :, -answer_len:, :].detach().cpu().numpy().tolist()
    save_attn = ctu.nested_list_processing(save_attn, func=lambda x: np.format_float_positional(x, precision=2))
    # save_dec_head_t = inp['decoder_input_ids'][0][-answer_len:]  # this is ground truth (should == answers_t if correct)
    save_head_tokens = ctu.decode_tokens(mt.tokenizer, answers_t)
    save_cand_tokens = [f'[{i}]' for i in range(mt.model.preseqlen)] + ctu.decode_tokens(mt.tokenizer, inp['input_ids'][0])

    result_dict['attentions'] = {
        # 'enc': {
        #     'attn': save_enc_attn,
        #     'head_tokens': save_enc_head_tokens,
        #     'cand_tokens': save_enc_cand_tokens,
        # },
        # 'cross': {
        #     'attn': save_cross_attn,
        #     'head_tokens': save_cross_head_tokens,
        #     'cand_tokens': save_cross_cand_tokens,
        # },
        # 'dec': {
        #     'attn': save_dec_attn,
        #     'head_tokens': save_dec_head_tokens,
        #     'cand_tokens': save_dec_cand_tokens,
        # }
        'all': {
            'attn': save_attn,
            'head_tokens': save_head_tokens,
            'cand_tokens': save_cand_tokens,
        }
    }

    return result_dict


def trace_exp4_1_attention_weights_distribution_gpt2(
    mt,
    a_ex,           # output from ctu.create_analysis_sample_dicts_all_nodes()
    # part='encoder',    # 'encoder', 'decoder', 'both'
    device='cuda',
):
    """
    AAA
    GPT2 Exp4.1 (inspect attention distribution for all encoder nodes, split by relevant (occ) vs. non-rel (non-occ) )
    """

    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    pre_sql_sequence = a_ex['pre_sql_sequence']
    # token_ranges_dict = a_ex['token_ranges_dict']     # Key doesn't exist, not sure why t5 version uses this
    token_ranges_dict = a_ex['struct_node_ranges_dict']
    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']
    text_st, text_ed = text_range
    struct_st, struct_ed = struct_range

    preseqlen = mt.model.preseqlen

    # expect_input_ranges = a_ex['expect_input_ranges']
    # tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    # expect = a_ex['expect']     # for funcs to work, not actually used here in 4.1

    inp = ctu2.make_inputs_gpt2(
        mt.tokenizer,
        # Here we study the counterpart of "Enc.SA", so no need for dec_prompt
        # However `enc_sentence` is not truncated, should probably use `pre_sql_sequence`
        [pre_sql_sequence],
        answer=None, 
        device=device
    )
    answer_len = 1      # for funcs to work, not actually used here in 4.1
    # if expect is not None:
    #     answer_len = len(mt.tokenizer.tokenize(expect))

    # trace_layers = [...]
    
    # ans_probs: (bsz, ans_len, vocab)
    ans_probs, attention_dict = trace_attention_uskg_gpt2(
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

    # attentions: layers, (bsz, n_heads, seq_len, prev_len + seq_len)
    attentions = attention_dict['attentions']
    attentions = torch.stack(attentions, dim=0).squeeze(dim=1)  # (layers, n_heads, seq_len, prev_len + seq_len)

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
            _attn = attentions[:, :, first_tok_pos, i]
            _sect_att_weights[f'prefix#{i}'] = _attn
        
        _text_attn = attentions[:, :, first_tok_pos, preseqlen + text_st : preseqlen + text_ed].sum(dim=-1)
        _sect_att_weights['text'] = _text_attn

        _self_ranges = a_ex['col_self_ranges'][col]
        _self_indices = [preseqlen + i for s, e in _self_ranges for i in range(s, e)]
        _self_attn = attentions[:, :, first_tok_pos, _self_indices].sum(dim=-1)
        _sect_att_weights['self'] = _self_attn

        ## context indices doesn't include EOS and mid prompt
        _context_ranges = a_ex['col_context_ranges'][col]
        _context_indices = [preseqlen + i for s, e in _context_ranges for i in range(s, e)]
        _context_attn = attentions[:, :, first_tok_pos, _context_indices].sum(dim=-1)
        _sect_att_weights['context'] = _context_attn

        ## ADD: "others" as a section (including EOS and mid prompt)
        _mid_attn = attentions[:, :, first_tok_pos, preseqlen + text_ed : preseqlen + struct_st].sum(dim=-1)
        # _eos_attn = attentions[:, :, first_tok_pos, -1]
        # _sect_att_weights['others'] = _mid_attn + _eos_attn
        _sect_att_weights['others'] = _mid_attn

        for k, v in _sect_att_weights.items():
            v = v.detach().cpu().numpy().tolist()
            _sect_att_weights[k] = ctu.nested_list_processing(v, func=lambda x: np.format_float_positional(x, precision=2))

        result_dict['attentions']['col'][col] = _sect_att_weights


    for tab in all_tabs:
        first_tok_pos = token_ranges_dict['table_name_ranges'][tab][0][0]

        _sect_att_weights = dict()
        for i in range(preseqlen):
            # size = (layers, n_heads); same below
            _attn = attentions[:, :, first_tok_pos, i]
            _sect_att_weights[f'prefix#{i}'] = _attn
        
        _text_attn = attentions[:, :, first_tok_pos, preseqlen + text_st : preseqlen + text_ed].sum(dim=-1)
        _sect_att_weights['text'] = _text_attn

        _self_ranges = a_ex['tab_self_ranges'][tab]
        _self_indices = [preseqlen + i for s, e in _self_ranges for i in range(s, e)]
        _self_attn = attentions[:, :, first_tok_pos, _self_indices].sum(dim=-1)
        _sect_att_weights['self'] = _self_attn

        ## context indices doesn't include EOS and mid prompt
        _context_ranges = a_ex['tab_context_ranges'][tab]
        _context_indices = [preseqlen + i for s, e in _context_ranges for i in range(s, e)]
        _context_attn = attentions[:, :, first_tok_pos, _context_indices].sum(dim=-1)
        _sect_att_weights['context'] = _context_attn

        ## ADD: "others" as a section (including EOS and mid prompt)
        _mid_attn = attentions[:, :, first_tok_pos, preseqlen + text_ed : preseqlen + struct_st].sum(dim=-1)
        # _eos_attn = attentions[:, :, first_tok_pos, -1]
        # _sect_att_weights['others'] = _mid_attn + _eos_attn
        _sect_att_weights['others'] = _mid_attn

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



def main_sdra_4_inspect_attention_gpt2(args):
    """
    Exp 4: Check the attention weights
    Purpose: directly look for information flow patterns
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    # exp_name = f'exp=4_{args.ds}_{args.subject_type}_{args.part}-tmp'
    # exp_name =  f'exp={args.exp_id}_{args.ds}' if args.is_on_syntax else f'exp={args.exp_id}_{args.ds}_{args.subject_type}'
    exp_name = f'exp={args.exp_id}_{args.ds}_{args.subject_type}'
    if args.is_tmp:
        exp_name += '-tmp'

    result_save_dir = os.path.join(args.result_dir, 'exp4_inspect_attention_gpt2')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg_gpt2 = ctu2.ModelAndTokenizer_USKG_GPT2('gpt2-medium-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg_gpt2)
    n_ex = len(processed_spider_dataset)

    total_samples = 0
    n_good_samples = 0

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 50
    stride = 111 if args.is_tmp else 10
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        # YS TODO (maybe): syntax
        analysis_samples = ctu2.create_analysis_sample_dicts_gpt2(
            mt_uskg_gpt2, ex, args.subject_type,
            remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
        # input_too_long = False
        # if len(analysis_samples) > 0:
        #     # enc_sentence = analysis_samples[0]['enc_sentence']
        #     # enc_tokenized = mt_uskg.tokenizer(enc_sentence)
        #     enc_tokenized = analysis_samples[0]['enc_tokenized']
        #     input_len = len(enc_tokenized['input_ids'])
        
        #     if input_len > 500:
        #         # ex_out_dict['trace_results'] = []
        #         ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
        #         input_too_long = True

        enc_sentence_toks = mt_uskg_gpt2.tokenizer.tokenize(ex['enc_sentence'])
        input_len = len(enc_sentence_toks)
        ## 362: max input len (excluding SQL) in our GPT2-medium config
        if input_len > 362:
            ex_out_dict['err_msg'] = f'Input too long: {input_len} > 362'
            print(f'* Warning: ex_id={ex_id}, Input too long: {input_len} > 362')
            continue

        ex_results = []
        for a_ex in analysis_samples:
            # if input_too_long:
            #     continue

            # a_ex = add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp4_inspect_attention_gpt2(mt_uskg_gpt2, a_ex)

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

    print(f'Total samples: {total_samples}')
    print(f'Good samples: {n_good_samples}')



def main_sdra_4_1_attention_weights_distribution(args):
    """
    Exp 4.1: Check the attention weights distribution for all nodes
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp={args.exp_id}_{args.ds}'
    if args.is_tmp:
        exp_name += '-tmp'

    result_save_dir = os.path.join(args.result_dir, 'exp4_1_attention_weights_distribution_gpt2')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg_gpt2 = ctu.ModelAndTokenizer_USKG_GPT2('gpt2-medium-prefix')

    processed_spider_dataset = ctu.load_spider_dataset(args, mt_uskg_gpt2)
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

        ex_out_dict = {'ex_id': ex_id}

        is_valid = True

        total_samples += 1

        try:
            a_ex = ctu2.create_analysis_sample_dicts_all_nodes_gpt2(
                mt_uskg_gpt2,
                ex,
                remove_struct_duplicate_nodes=True)
            
            # enc_tokenized = a_ex['enc_tokenized']
            # input_len = len(enc_tokenized['input_ids'])
            enc_sentence_toks = mt_uskg_gpt2.tokenizer.tokenize(ex['enc_sentence'])
            input_len = len(enc_sentence_toks)
            ## 362: max input len (excluding SQL) in our GPT2-medium config
            if input_len > 362:
                print(f'* Warning: ex_id={ex_id}, Input too long: {input_len} > 362')
                raise ctu.SDRASampleError(f'Input too long: {input_len} > 362')
            
            result = trace_exp4_1_attention_weights_distribution_gpt2(mt_uskg_gpt2, a_ex)

        except ctu.SDRASampleError as e:
            ex_out_dict['err_msg'] = str(e)

            ## Make some placeholders 
            # result_dict = ctu.make_basic_result_dict(ex)
            # result_dict['attentions'] = {
            #     'col': dict(),
            #     'tab': dict(),
            # }
            ex_out_dict['trace_results'] = None

            is_valid = False
        
        if is_valid:
            ex_out_dict['trace_results'] = result
            n_good_samples += 1

        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
        f.flush()
    f.close()

    print(f'Total samples: {total_samples}')
    print(f'Good samples: {n_good_samples}')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_id', required=True, help='Experiment ID')
    parser.add_argument('-t', '--is_tmp', action='store_true', help='Do a temp debug run')
    parser.add_argument('-ds', '--ds', default='dev', help='The data split to use: dev | train')

    args = parser.parse_args()

    # args.subject_type = 'column'         # table, table_alias, column, (value)
    # args.part = 'both'               # encoder, decoder, both
    args.spider_dataset_path = f'/home/yshao/Projects/SDR-analysis/data/spider/{args.ds}+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'
    args.spider_tables_path = '/home/yshao/Projects/language/language/xsp/data/spider/tables.json'

    args.result_dir = '/home/yshao/Projects/rome/results/gpt2_tracing'

    ctu.evaluate_hardness.evaluator = ctu.load_evaluator(args)

    if args.exp_id == '4':
        for subject_type in ['column', 'table', 'table_alias']:
            args.subject_type = subject_type
            main_sdra_4_inspect_attention_gpt2(args)

    elif args.exp_id == '4.1':
        main_sdra_4_1_attention_weights_distribution(args)

    else:
        raise ValueError(args.exp_id)


if __name__ == "__main__":
    main()
