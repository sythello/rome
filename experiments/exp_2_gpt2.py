import argparse
from argparse import Namespace
import json
import os
import sys
import re
from collections import defaultdict, Counter
import copy

import numpy
import torch
from torch import nn
from datasets import load_dataset
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from util.uskg import USKG_SPLITTER, USKG_SPLITTER_CHARS, RAT_SQL_RELATION_ID2NAME, \
    SQL_SYNTAX_PHRASES, SQL_SYNTAX_PUNCTS, SDRASampleError, \
    load_model_uskg, load_raw_dataset, load_spider_dataset, run_model_forward_uskg, \
    decode_sentences, decode_tokens, find_token_range, \
    find_text_struct_in_range, find_text_struct_in_range_str_tokens, find_struct_name_ranges, \
    separate_punct, separate_punct_by_offset, categorize_tokens_offset, \
    make_dec_prompt, parse_struct_in, ensure_list, \
    ModelAndTokenizer_USKG, layername_uskg, load_evaluator, \
    evaluate_hardness, detect_node_role, check_col_text_match, check_table_text_match, \
    parse_sql_alias2table, nested_list_processing, nested_json_processing

from util.uskg_gpt2 import ModelAndTokenizer_USKG_GPT2, run_model_forward_uskg_gpt2, layername_uskg_gpt2

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

from experiments import causal_trace_uskg as ctu
from experiments import causal_trace_uskg_gpt2 as ctu2



# def trace_section_corrupt_restore(
#     mt,
#     ex,
#     subject_type='column',
#     corrupt_section='text', # 'text', 'struct'
#     samples=10,
#     noise=0.1,
#     part='encoder',         # For now, only 'encoder' supported for section corrupt
#     uniform_noise=False,
#     replace=True,
#     window=10,
#     kind=None,
#     remove_struct_duplicate_nodes=True,     # If True, remove column names appearing multiple times in struct
# ):
#     """
#     AAA
#     Exp 2.2
#     TODO: modify
#     """

#     if part != 'encoder':
#         raise ValueError(part)

#     # token_ranges_dict = find_struct_name_ranges(mt.tokenizer, ex)
    
#     analysis_samples = create_analysis_sample_dicts(mt, ex, subject_type, remove_struct_duplicate_nodes)

#     all_results = []
#     for a_ex in analysis_samples:
#         # tok_ranges = node_name_ranges[node]
#         # TODO (later): handle duplicate cols in struct_in
#         # full token range as a single item, to restore simultaneously

#         # text_range, struct_range = find_text_struct_in_range(mt.tokenizer, enc_tokenized['input_ids'])
#         text_range = a_ex['text_range']
#         struct_range = a_ex['struct_range']
#         if corrupt_section == 'text':
#             corrupt_tok_indices = list(range(*text_range))
#         elif corrupt_section == 'struct':
#             corrupt_tok_indices = list(range(*struct_range))
#         else:
#             raise ValueError(corrupt_section)

#         expect_input_ranges = a_ex['expect_input_ranges']
#         tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
#         enc_sentence = a_ex['enc_sentence']
#         dec_prompt = a_ex['dec_prompt']
#         node = a_ex['expect']

#         enc_token_range = [[i for s, e in expect_input_ranges for i in range(s, e)]]
#         dec_token_range = []

#         trace_result = calculate_hidden_flow_uskg(
#             mt,
#             enc_sentence=enc_sentence,
#             dec_prompt=dec_prompt,
#             expect=node,
#             e_range=corrupt_tok_indices,
#             tokens_to_mix_individual_indices=True,
#             samples=samples,
#             noise=noise,
#             # token_range=token_range,
#             enc_token_range=enc_token_range,
#             dec_token_range=dec_token_range,
#             uniform_noise=uniform_noise,
#             replace=replace,
#             window=window,
#             kind=kind,
#         )
        
#         result = make_basic_result_dict(a_ex)
#         for k, v in trace_result.items():
#             if k not in result:
#                 result[k] = v
#             else:
#                 assert result[k] == v, (result[k], v)
#         result['part'] = part
#         all_results.append(result)
#     return all_results


def trace_exp2_section_corrupt_restore_gpt2(
    mt,
    a_ex,
):
    """
    Exp2 for GPT2
    ex (Dict): analysis_sample (a_ex) with `add_clean_prediction()`
    """

    # text_in = a_ex['text_in']
    # struct_in = a_ex['struct_in']
    # enc_sentence = a_ex['enc_sentence']
    # enc_tokenized = a_ex['enc_tokenized']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']
    text_st, text_ed = text_range
    struct_st, struct_ed = struct_range
    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))

    self_w_bound_ranges = a_ex['self_ranges']
    self_no_bound_ranges = a_ex['expect_input_ranges']
    struct_context_no_bound_ranges = a_ex['context_ranges']

    self_w_bound_tok_indices = [i for s, e in self_w_bound_ranges for i in range(s, e)]
    self_no_bound_tok_indices = [i for s, e in self_no_bound_ranges for i in range(s, e)]
    struct_context_w_bound_tok_indices = [tnum for tnum in range(*struct_range) if tnum not in self_no_bound_tok_indices]
    struct_context_no_bound_tok_indices = [i for s, e in struct_context_no_bound_ranges for i in range(s, e)]

    expect = a_ex['expect']
    # dec_prompt = make_dec_prompt(ex['seq_out'], expect)
    dec_prompt = a_ex['dec_prompt']

    parsed_struct_in = a_ex['parsed_struct_in']
    col2table = a_ex['col2table']
    node_name_ranges = a_ex['node_name_ranges']
    subject_type = a_ex['expect_type']

    expect_input_ranges = a_ex['expect_input_ranges']
    if len(expect_input_ranges) > 1:
        assert not a_ex['remove_struct_duplicate_nodes']
        # raise NotImplementedError
    
    # node_range, = expect_input_ranges
    # struct_no_node_toks = [tnum for tnum in range(*struct_range) if tnum not in range(*node_range)]
    
    result = ctu.make_basic_result_dict(a_ex)
    result['trace_scores'] = {
        'corrupt': dict(),
        'corrupt_text_restore': dict(),
    }

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = a_ex['base_score']
    is_correct_pred = result['correct_prediction']

    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    # if base_score < 0.5:
    if answer.strip() != expect:
        assert False, "Incorrect prediction should already been skipped!"
    #     n_too_hard += 1
    #     result['is_good_sample'] = False
    #     ex_results.append(result)
    #     continue

    corrupt_section_tok_indices_dict = {
        'text': text_tok_indices,
        'struct': struct_tok_indices,
        'self': self_no_bound_tok_indices,
        'struct_context': struct_context_no_bound_tok_indices,
        # 'other': other_tok_indices,
        # 'text+other': text_tok_indices + other_tok_indices,
        'all': list(range(struct_ed)),
    }

    restore_section_tok_indices_dict = {
        'text': text_tok_indices,
        'struct': struct_tok_indices,
        'self': self_w_bound_tok_indices,
        'struct_context': struct_context_w_bound_tok_indices,
        # 'other': other_tok_indices,
        # 'text+other': text_tok_indices + other_tok_indices,
        'all': list(range(struct_ed)),
    }

    layer_names_dict = {
        'embed': ctu2.layername_uskg_gpt2(mt.model, 0, 'embed'),
        'mid': ctu2.layername_uskg_gpt2(mt.model, mt.num_layers // 2),
    }

    # print('* corrupt_section_tok_indices_dict:', corrupt_section_tok_indices_dict)
    # print('* restore_section_tok_indices_dict:', restore_section_tok_indices_dict)
    # print('* layer_names_dict', layer_names_dict)

    inp = ctu2.make_inputs_gpt2(
        mt.tokenizer,
        [dec_prompt] * 2,
        answer=expect)

    # encoder_text_mid_layer_states = [
    #     (tnum, ctu2.layername_uskg_gpt2(mt.model, mt.num_layers - 1))
    #     for tnum in range(*text_range)
    # ]
    # encoder_struct_mid_layer_states = [
    #     (tnum, ctu2.layername_uskg_gpt2(mt.model, mt.num_layers - 1))
    #     for tnum in range(*struct_range)
    # ]
    # encoder_node_mid_layer_states = [
    #     (tnum, ctu2.layername_uskg_gpt2(mt.model, mt.num_layers - 1))
    #     for tnum in range(*node_range)
    # ]
    # encoder_struct_no_node_mid_layer_states = [
    #     (tnum, ctu2.layername_uskg_gpt2(mt.model, mt.num_layers - 1))
    #     for tnum in struct_no_node_toks
    # ]

    # answer_len = len(mt.tokenizer.tokenize(expect))
    # answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]
    # base_score = min(base_score).item()
    # answer = decode_sentences(mt.tokenizer, answers_t)
    
    """ Starting Exp2"""
    result['base_score'] = base_score

    # print("* corrupt_section_tok_indices_dict['all']:", corrupt_section_tok_indices_dict['all'])
    # print("* layer_names_dict['embed']:", layer_names_dict['embed'])

    low_score = result['low_score'] = ctu2.trace_with_repatch_uskg_gpt2(
        model=mt.model,
        inp=inp,
        states_to_patch=[],
        states_to_unpatch=[],
        answers_t=answers_t,
        states_to_corrupt=[(t, layer_names_dict['embed']) for t in corrupt_section_tok_indices_dict['all']],
        noise=0.0,
        replace=True,
    ).item()
    
    ## Corrupting text: expect wrong pred 
    if base_score - low_score < 0.5:
        # n_too_easy += 1
        result['is_good_sample'] = False
        return result
    
    # n_good_samples += 1
    result['is_good_sample'] = True
    
    for sect_k, sect_tok_indices in corrupt_section_tok_indices_dict.items():
        result['trace_scores']['corrupt'][sect_k] = ctu2.trace_with_repatch_uskg_gpt2(
            model=mt.model,
            inp=inp,
            states_to_patch=[],
            states_to_unpatch=[],
            answers_t=answers_t,
            states_to_corrupt=[(t, layer_names_dict['embed']) for t in sect_tok_indices],
            noise=0.0,
            replace=True,
        ).item()

        result['trace_scores']['corrupt_text_restore'][sect_k] = ctu2.trace_with_repatch_uskg_gpt2(
            model=mt.model,
            inp=inp,
            states_to_patch=[(t, layer_names_dict['mid']) for t in restore_section_tok_indices_dict[sect_k]],
            states_to_unpatch=[],
            answers_t=answers_t,
            states_to_corrupt=[(t, layer_names_dict['embed']) for t in corrupt_section_tok_indices_dict['text']],
            noise=0.0,
            replace=True,
        ).item()


    # ## Restoring text encoding for decoder, but struct encoding are with dirty text encoding 
    # r_text_score = ctu2.trace_with_repatch_uskg_gpt2(
    #     model=mt.model,
    #     inp=inp,
    #     states_to_patch=encoder_text_mid_layer_states,
    #     states_to_unpatch=[],
    #     answers_t=answers_t,
    #     tokens_to_mix=text_range,
    #     tokens_to_mix_individual_indices=False,
    #     replace=True,
    # ).item()
    
    # ## Restoring clean struct encoding but dirty text encoding for decoder
    # r_struct_score = ctu2.trace_with_repatch_uskg_gpt2(
    #     model=mt.model,
    #     inp=inp,
    #     states_to_patch=encoder_struct_mid_layer_states,
    #     states_to_unpatch=[],
    #     answers_t=answers_t,
    #     tokens_to_mix=text_range,
    #     tokens_to_mix_individual_indices=False,
    #     replace=True,
    # ).item()
    
    # ## Restoring clean node_name encoding but dirty text encoding for decoder (stricter than above)
    # r_node_score = ctu2.trace_with_repatch_uskg_gpt2(
    #     model=mt.model,
    #     inp=inp,
    #     states_to_patch=encoder_node_mid_layer_states,
    #     states_to_unpatch=[],
    #     answers_t=answers_t,
    #     tokens_to_mix=text_range,
    #     tokens_to_mix_individual_indices=False,
    #     replace=True,
    # ).item()

    # ## Restoring struct except column of interest. Check contextualization of this column into other nodes
    # r_struct_no_node_score = ctu2.trace_with_repatch_uskg_gpt2(
    #     model=mt.model,
    #     inp=inp,
    #     states_to_patch=encoder_struct_no_node_mid_layer_states,
    #     states_to_unpatch=[],
    #     answers_t=answers_t,
    #     tokens_to_mix=text_range,
    #     tokens_to_mix_individual_indices=False,
    #     replace=True,
    # ).item()

    # ## Restoring clean col_name encoding; corrupt all tokens. Check if only this column is enough 
    # r_node_corrupt_all_score = trace_with_repatch_uskg(
    #     model=mt.model,
    #     inp=inp,
    #     states_to_patch=encoder_node_last_layer_states,
    #     states_to_unpatch=[],
    #     answers_t=answers_t,
    #     tokens_to_mix=(0, struct_range[1]),
    #     tokens_to_mix_individual_indices=False,
    #     replace=True,
    # ).item()
    
    # result['trace_scores']['r_text'] = r_text_score
    # result['trace_scores']['r_struct'] = r_struct_score
    # result['trace_scores']['r_node'] = r_node_score
    # result['trace_scores']['r_struct_no_node'] = r_struct_no_node_score
    return result


# def main_sdra_exp2_section_corrupt_restore_gpt2(args):
#     """
#     AAA
#     Exp 2 for GPT2
#     """
#     spider_dataset_path = args.spider_dataset_path
#     spider_db_dir = args.spider_db_dir
#     data_cache_dir = args.data_cache_dir

#     exp_name = f'exp=2_{args.ds}_{args.subject_type}'
#     result_save_dir = os.path.join(args.result_dir, 'exp2.2_dirty_text_struct_restore')
#     os.makedirs(result_save_dir, exist_ok=True)
#     result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

#     mt_uskg = ModelAndTokenizer_USKG_GPT2('gpt2-medium-prefix')

#     processed_spider_dataset = load_spider_dataset(args, mt_uskg)
#     n_ex = len(processed_spider_dataset)

#     start_id = 0
#     end_id = n_ex
#     stride = 1
#     with open(result_save_path, 'w') as f:
#         for i in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
#             ex = processed_spider_dataset[i]
#             results = trace_exp2_section_corrupt_restore_gpt2(
#                 mt=mt_uskg,
#                 ex=ex,
#                 subject_type=args.subject_type,
#                 replace=True,
#                 # part=args.part,
#                 # part='encoder'
#             )
#             dump_dict = dict(
#                 ex_id=i,
#                 trace_results=results,
#             )
#             f.write(json.dumps(dump_dict, indent=None) + '\n')


def main_sdra_exp2_section_corrupt_restore_gpt2(args):
    """
    Exp2 for GPT2
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_config = EXP_CONFIGS[args.exp_id]
    args.replace = exp_config['replace']
    args.noise = exp_config['noise']
    args.is_on_syntax = (args.subject_type == 'non_node')

    exp_name =  f'exp={args.exp_id}_{args.ds}' if args.is_on_syntax else f'exp={args.exp_id}_{args.ds}_{args.subject_type}'
    exp_name += f'-replace={args.replace}-noise={args.noise}'
    if args.is_tmp:
        exp_name += '-tmp'

    result_save_dir = args.result_save_dir
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg_gpt2 = ModelAndTokenizer_USKG_GPT2('gpt2-medium-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg_gpt2)
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

        if args.is_on_syntax:
            raise NotImplementedError
            # analysis_samples = create_syntax_analysis_sample_dicts(mt_uskg_gpt2, ex)
        else:
            analysis_samples = ctu2.create_analysis_sample_dicts_gpt2(
                mt_uskg_gpt2, ex, args.subject_type,
                remove_struct_duplicate_nodes=True)

        ex_out_dict = {'ex_id': ex_id}
        
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

            a_ex = ctu2.add_clean_prediction_gpt2(mt_uskg_gpt2, a_ex)
            
            # exp_func = trace_exp2_3
            result = trace_exp2_section_corrupt_restore_gpt2(
                mt_uskg_gpt2,
                a_ex,
                # replace=args.replace,
                # noise=args.noise,
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

    print(f'Total samples: {total_samples}')
    print(f'Good samples: {n_good_samples}')


EXP_CONFIGS = {
    'gpt2_2.0': {
        'replace': True,
        'noise': 0.0,
    },
    # '2.3.0': {
    #     'replace': True,
    #     'noise': 0.1,
    # },
    # '2.3.1': {
    #     'replace': True,
    #     'noise': 0.0,
    # },
    # '2.3.2': {
    #     'replace': False,
    #     'noise': 0.1,
    # },
}


def main():
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add different types of arguments
    parser.add_argument('-e', '--exp_id', required=False, default='gpt2_2.0', help='Experiment ID')
    parser.add_argument('-t', '--is_tmp', action='store_true', help='Do a temp debug run')
    # parser.add_argument('-d', '--arg4', choices=['option1', 'option2', 'option3'], help='An argument with limited choices')
    # parser.add_argument('-e', '--arg5', nargs='+', help='An argument with multiple values')

    # Parse the command-line arguments
    args = parser.parse_args()

    # args = Namespace()
    args.ds = 'dev'                     # train, dev
    # args.subject_type = 'column'         # table, table_alias, column, (value)
    # args.part = 'encoder'               # encoder, decoder, both
    args.spider_dataset_path = f'/home/yshao/Projects/SDR-analysis/data/spider/{args.ds}+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'
    args.spider_tables_path = '/home/yshao/Projects/language/language/xsp/data/spider/tables.json'

    args.result_dir = '/home/yshao/Projects/rome/results'

    evaluate_hardness.evaluator = load_evaluator(args)

    args.result_save_dir = os.path.join(args.result_dir, "gpt2_tracing", "exp2_section_corrupt_restore_gpt2")

    args.subject_type = 'column'
    main_sdra_exp2_section_corrupt_restore_gpt2(args)

    args.subject_type = 'table'
    main_sdra_exp2_section_corrupt_restore_gpt2(args)

    args.subject_type = 'table_alias'
    main_sdra_exp2_section_corrupt_restore_gpt2(args)


if __name__ == "__main__":
    main()
