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


def trace_exp6_0(
    mt,
    ex,
):
    """
    Exp6.0: encoding-corruption-effect-syntax
    Corrupting input embedding / final encoding of different input sections - on syntax token predictions (now also include literals).
    Sections: text, struct, cols, tables
    Purpose: check the necessity of each section for syntax prediction. Also, prepare for operational items.
    """

    text_in = ex['text_in']
    struct_in = ex['struct_in']
    enc_sentence = ex['enc_sentence']
    enc_tokenized = ex['enc_tokenized']
    text_range = ex['text_range']
    struct_range = ex['struct_range']

    expect = ex['expect']
    # dec_prompt = make_dec_prompt(ex['seq_out'], expect)
    dec_prompt = ex['dec_prompt']

    parsed_struct_in = ex['parsed_struct_in']
    col2table = ex['col2table']
    node_name_ranges = ex['node_name_ranges']
    subject_type = ex['expect_type']
    expect_input_ranges = ex['expect_input_ranges']

    if len(expect_input_ranges) > 1:
        assert not ex['remove_struct_duplicate_nodes']
        raise NotImplementedError
    
    node_range, = expect_input_ranges

    struct_no_node_toks = [tnum for tnum in range(*struct_range) if tnum not in range(*node_range)]
    
    result = make_basic_result_dict(ex)
    result['mutual_scores'] = dict()

    inp = make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * 11,
        [dec_prompt] * 11,
        answer=expect)

    encoder_text_last_layer_states = [
        (tnum, layername_uskg(mt.model, 'encoder', mt.num_enc_layers - 1))
        for tnum in range(*text_range)
    ]
    encoder_struct_last_layer_states = [
        (tnum, layername_uskg(mt.model, 'encoder', mt.num_enc_layers - 1))
        for tnum in range(*struct_range)
    ]
    encoder_node_last_layer_states = [
        (tnum, layername_uskg(mt.model, 'encoder', mt.num_enc_layers - 1))
        for tnum in range(*node_range)
    ]
    encoder_struct_no_node_last_layer_states = [
        (tnum, layername_uskg(mt.model, 'encoder', mt.num_enc_layers - 1))
        for tnum in struct_no_node_toks
    ]

    # answer_len = len(mt.tokenizer.tokenize(expect))
    # answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]
    # base_score = min(base_score).item()
    # answer = decode_sentences(mt.tokenizer, answers_t)
    
    answer = result['answer']
    answers_t = result['answers_t']
    base_score = ex['base_score']
    is_correct_pred = result['correct_prediction']
    
    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    """ Starting Exp2.1.1: finer grained text struct mutual """
    result['mutual_scores']['clean_t-clean_s'] = base_score
    
    # equivalent to restore_text
    result['mutual_scores']['clean_t-dc_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_text_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=text_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    result['mutual_scores']['clean_t-dirty_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_text_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=struct_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    result['mutual_scores']['dc_t-clean_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_struct_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=struct_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    ## For exp-2.1: mutual corruption
    # First pass: corrupt text (no restore)
    # Second pass: corrupt struct, no restore, reset struct output to first pass
    result['mutual_scores']['dc_t-dc_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=[],
        states_to_unpatch=encoder_struct_last_layer_states,
        answers_t=answers_t,
        tokens_to_mix_1st_pass=text_range,
        tokens_to_mix=struct_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    result['mutual_scores']['dc_t-dirty_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=[],
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=struct_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    # equivalent to restore_struct
    result['mutual_scores']['dirty_t-clean_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_struct_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=text_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    # equivalent to low_score (for text corruption)
    result['mutual_scores']['dirty_t-dc_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=[],
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=text_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    result['mutual_scores']['dirty_t-dirty_s'] = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=[],
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=list(range(*text_range)) + list(range(*struct_range)),
        tokens_to_mix_individual_indices=True,
        replace=True,
    ).item()
    
    # for k in mutual_scores_dict:
    #     mutual_scores_dict[k].append(result['mutual_scores'][k])
    
    """ Starting Exp2: dirty text recovery """
    result['base_score'] = result['mutual_scores']['clean_t-clean_s']
    low_score = result['low_score'] = result['mutual_scores']['dirty_t-dc_s']
    # if base_score < 0.5:
    if answer.strip() != expect:
        assert False, "Incorrect prediction should already been skipped!"
    #     n_too_hard += 1
    #     result['is_good_sample'] = False
    #     ex_results.append(result)
    #     continue
    
    ## Corrupting text: expect wrong pred 
    if base_score - low_score < 0.5:
        # n_too_easy += 1
        result['is_good_sample'] = False
        return result
    
    # n_good_samples += 1
    result['is_good_sample'] = True
    
    ## Restoring text encoding for decoder, but struct encoding are with dirty text encoding 
    r_text_score = result['mutual_scores']['clean_t-dc_s']
    
    ## Restoring clean struct encoding but dirty text encoding for decoder
    r_struct_score = result['mutual_scores']['dirty_t-clean_s']
    
    ## Restoring clean node_name encoding but dirty text encoding for decoder (stricter than above)
    r_node_score = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_node_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=text_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()

    ## Restoring struct except column of interest. Check contextualization of this column into other nodes
    r_struct_no_node_score = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_struct_no_node_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=text_range,
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()

    ## Restoring clean col_name encoding; corrupt all tokens. Check if only this column is enough 
    r_node_corrupt_all_score = trace_with_repatch_uskg(
        model=mt.model,
        inp=inp,
        states_to_patch=encoder_node_last_layer_states,
        states_to_unpatch=[],
        answers_t=answers_t,
        tokens_to_mix=(0, struct_range[1]),
        tokens_to_mix_individual_indices=False,
        replace=True,
    ).item()
    
    result['r_text_score'] = r_text_score
    result['r_struct_score'] = r_struct_score
    result['r_node_score'] = r_node_score
    result['r_struct_no_node_score'] = r_struct_no_node_score
    result['r_node_corrupt_all_score'] = r_node_corrupt_all_score
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
