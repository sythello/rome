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



def trace_exp7_0_0(
    mt,
    a_ex,                   # output from ctu.create_analysis_sample_dicts()
    device='cuda',
):
    """
    AAAA
    Exp7.0.0 (encoder-layer-skip-effect)
    Skip certain layers and check the effect
    """

    part = 'encoder'

    # expect_input_ranges = a_ex['expect_input_ranges']
    # tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]
    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']
    text_st, text_ed = text_range
    struct_st, struct_ed = struct_range
    text_tok_indices = list(range(*text_range))
    struct_tok_indices = list(range(*struct_range))
    # prefix_len = mt.model.preseqlen

    self_ranges = a_ex['self_ranges']
    struct_context_ranges = a_ex['context_ranges']
    self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
    struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]

    result = ctu.make_basic_result_dict(a_ex)

    ## Check basic results
    n_samples = 2       # layer copy doesn't have randomness 

    inp = ctu.make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * n_samples,
        [dec_prompt] * n_samples,
        answer=expect,
        device=device
    )

    answer = result['answer']
    answers_t = result['answers_t']
    base_score = result['base_score']
    is_correct_pred = result['correct_prediction']

    bs, enc_seq_len = inp['input_ids'].size()
    bs, dec_seq_len = inp['decoder_input_ids'].size()

    other_tok_indices = list(range(text_ed, struct_st)) + list(range(struct_ed, enc_seq_len))
    

    if not is_correct_pred:
        # scores don't make sense when clean pred is wrong 
        result['is_good_sample'] = False
        return result
    
    """ Low score of exp 7: section = all, layer = 0 to 23 """

    N_layers = mt.num_enc_layers if part == 'encoder' else mt.num_dec_layers
    sublayers = ['self_attn', 'mlp'] if part == 'encoder' else ['self_attn', 'cross_attn', 'mlp']

    section_tok_indices_dict = {
        'text': text_tok_indices,
        'struct': struct_tok_indices,
        'self': self_tok_indices,
        'struct_context': struct_context_tok_indices,
        'other': other_tok_indices,
        'text+other': text_tok_indices + other_tok_indices,
        'all': list(range(enc_seq_len)),
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

    # corrupted_vocab_probs: (answer_len, vocab_size)
    corrupted_vocab_probs = ctu.run_repatch_uskg_multi_token(
        model=mt.model,
        inp=inp,
        answer_len=len(answers_t),
        states_to_patch=[],
        states_to_unpatch=[],
        states_to_corrupt=[(t, ctu.layername_uskg(mt.model, part, l, sublayer))
                           for t in section_tok_indices_dict['all'] for l in range(N_layers) for sublayer in sublayers],
        # mix_mask_per_layer={ctu.layername_uskg(mt.model, part, l, attn_type) : att_mix_mask_dict['all'] for l in range(N_layers)},
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

    ## If base and low score has no diff, "too easy", skip
    if base_score - low_score < 0.5:
        result['is_good_sample'] = False
        return result

    result['is_good_sample'] = True
    
    """ Starting Exp7.0.0 """


    result['trace_scores'] = dict()

    for sect_k, sect_tok_indices in section_tok_indices_dict.items():
        # TEMP: only run for newly added sections
        # if not sect_k.endswith('o'):
        #     continue
        # END TEMP

        result['trace_scores'][sect_k] = dict()

        for layers_k, layers_range in layers_range_dict.items():
            _score = ctu.trace_with_repatch_uskg(
                model=mt.model,
                inp=inp,
                answers_t=answers_t,
                states_to_patch=[],
                states_to_unpatch=[],
                states_to_corrupt=[(t, ctu.layername_uskg(mt.model, part, l, sublayer))
                                for t in sect_tok_indices for l in layers_range for sublayer in sublayers],
                replace=True,
            ).item()
            result['trace_scores'][sect_k][layers_k] = _score

    return result


def main_sdra_7_0_0_encoder_layer_skip_effect(args):
    """
    Exp 7.0.0: Skip certain layers and check the effect
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=7.0.0_{args.ds}_{args.subject_type}'
    result_save_dir = os.path.join(args.result_dir, 'exp7_0_0_encoder_layer_skip_effect')
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
            
            result = trace_exp7_0_0(
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



def main():
    args = Namespace()
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
    main_sdra_7_0_0_encoder_layer_skip_effect(args)

    args.subject_type = 'table'
    main_sdra_7_0_0_encoder_layer_skip_effect(args)

    args.subject_type = 'table_alias'
    main_sdra_7_0_0_encoder_layer_skip_effect(args)


if __name__ == "__main__":
    main()

