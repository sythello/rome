import argparse
from argparse import Namespace
import json
import os
import sys
import re
from collections import defaultdict
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
from util.uskg import USKG_SPLITTER, USKG_SPLITTER_CHARS, load_model_uskg, load_raw_dataset, run_model_forward_uskg, \
    decode_sentences, decode_tokens, find_token_range, find_text_struct_in_range, find_struct_name_ranges, \
    separate_punct, make_dec_prompt, parse_struct_in, ensure_list

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


def main():
    args = Namespace()
    args.subject_type = 'column'         # table, column, value
    args.part = 'encoder'               # encoder, decoder, both
    args.spider_dev_path = '/home/yshao/Projects/SDR-analysis/data/spider/dev+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'

    args.result_dir = '/home/yshao/Projects/rome/results'

    main_sdra_2_2_dirty_text_struct_restore(args)


def main_sdra_1_struct_node_restore(args):
    """
    Exp 1: Corrupt the struct node of interest, restore every single state
    Purpose: check where is the node-relevant info stored
    """
    spider_dev_path = args.spider_dev_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'dev_{args.subject_type}_{args.part}'
    result_save_dir = os.path.join(args.result_dir, 'struct_node_restore')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    raw_spider_dev = load_raw_dataset(
        data_filepath = spider_dev_path,
        db_path=spider_db_dir,
    )
    processed_spider_dev = s2s_spider.DevDataset(
        args=mt_uskg.task_args,
        raw_datasets=raw_spider_dev,
        cache_root=data_cache_dir
    )
    n_ex = len(processed_spider_dev)

    with open(result_save_path, 'a') as f:
        for i in tqdm(range(0, n_ex), desc=f"Main loop: {exp_name}", ascii=True):
            ex = processed_spider_dev[i]
            results = trace_struct_restore(
                mt=mt_uskg,
                ex=ex,
                subject_type=args.subject_type,
                replace=True,
                part=args.part,
            )
            dump_dict = dict(
                ex_id=i,
                trace_results=results,
            )
            f.write(json.dumps(dump_dict, indent=None) + '\n')


def main_sdra_2_2_dirty_text_struct_restore(args):
    """
    Exp 2.2: Corrupt the text, restoring the struct node of interest
    Purpose: check the pos of contextualization of text into struct nodes
    """
    spider_dev_path = args.spider_dev_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=2.2_dev_{args.subject_type}'
    result_save_dir = os.path.join(args.result_dir, 'dirty_text_struct_restore')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    raw_spider_dev = load_raw_dataset(
        data_filepath = spider_dev_path,
        db_path=spider_db_dir,
    )
    processed_spider_dev = s2s_spider.DevDataset(
        args=mt_uskg.task_args,
        raw_datasets=raw_spider_dev,
        cache_root=data_cache_dir
    )
    n_ex = len(processed_spider_dev)

    with open(result_save_path, 'a') as f:
        for i in tqdm(range(0, n_ex), desc=f"MAIN: {exp_name}", ascii=True):
            ex = processed_spider_dev[i]
            results = trace_section_corrupt_restore(
                mt=mt_uskg,
                ex=ex,
                subject_type=args.subject_type,
                replace=True,
                # part=args.part,
                part='encoder'
            )
            dump_dict = dict(
                ex_id=i,
                trace_results=results,
            )
            f.write(json.dumps(dump_dict, indent=None) + '\n')


def trace_with_patch_uskg_multi_token(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end) (encoder input)
    tokens_to_mix_individual_indices=False,     # If False (default), `tokens_to_mix` is a range; if True, `tokens_to_mix` is a list of indices
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.

    (YS) For now, assuming only corrupting encoder input sentence.
    The decoder_input has been changed to include the correct
    answer (excluding the last token), and need to get the probability and
    correctness of the last N tokens (N is length of answers_t). Return
    the minimum prob of these tokens.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    # print('* patch_spec:', patch_spec)

    embed_layername = layername_uskg(model, "encoder", 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    if tokens_to_mix is None:
        tokens_to_mix_indices = None
    elif tokens_to_mix_individual_indices:
        tokens_to_mix_indices = tokens_to_mix
    else:
        tokens_to_mix_indices = list(range(*tokens_to_mix))

    def patch_rep(x, layer):
        # x: original layer output; Return: maybe modified layer output
        # x: (bsize, tokens, ...)
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                # print('** layer:', layer)
                # print('** x.shape:', x.shape)
                # print('** tokens_to_mix:', tokens_to_mix)

                # b, e = tokens_to_mix
                mix_len = len(tokens_to_mix_indices)
                noise_data = noise_fn(
                    # torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                    torch.from_numpy(prng(x.shape[0] - 1, mix_len, x.shape[2]))
                ).to(device=x.device, dtype=x.dtype)
                # print(tokens_to_mix_indices)
                # print('x:', x.dtype)
                # print('noise_data:', noise_data.dtype)
                if replace:
                    x[1:, tokens_to_mix_indices] = noise_data
                else:
                    x[1:, tokens_to_mix_indices] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        # print('** x:', type(x))
        # print('** layer:', layer)
        # print('** h.size():', h.size())
        # print('** tokens to restore:', patch_spec[layer])
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        # outputs_exp = model(**inp)
        outputs_exp = run_model_forward_uskg(model, **inp)
        # YS debug
        # outputs_exp = Namespace()
        # outputs_exp.logits = torch.randn(
        #     inp['decoder_input_ids'].size(0),                           # bsize
        #     inp['decoder_input_ids'].size(1),                           # seq_len
        #     model.pretrain_model.encoder.embed_tokens.weight.size(0)    # vocab_size
        # )
        # end debug

    # print(outputs_exp.logits.size())
    # We report softmax probabilities for the answers_t token predictions of interest.
    # (YS) allowing multi-token answers_t
    all_ans_probs = []
    try:
        _ = answers_t[0]
    except:
        # not sequence type, make it a list
        answers_t = [answers_t]
    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        probs = torch.softmax(outputs_exp.logits[1:, -len(answers_t) + i, :], dim=1).mean(dim=0)[_t]
        # probs = probs.detach().cpu().numpy().round(decimals=4).to_list()
        all_ans_probs.append(probs)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return all_ans_probs, all_traced

    return all_ans_probs


def trace_with_patch_uskg(*args, **kwargs):
    """ For compatibility, make this a wrapper for trace_with_patch_uskg_multi_token """

    ret = trace_with_patch_uskg_multi_token(*args, **kwargs)
    if isinstance(ret, tuple):
        all_ans_probs, all_traced = ret
        probs = min(all_ans_probs)
        return probs, all_traced
    else:
        all_ans_probs = ret
        probs = min(all_ans_probs)
        return probs


def trace_with_repatch_uskg_multi_token(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,    # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,      # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    states_to_patch_1st_pass=list(),    # states to restore for the 1st pass (default: empty)
    tokens_to_mix_1st_pass=None,        # tokens to corrupt in the 1st pass (default: None)
    tokens_to_mix_individual_indices=False,     # If False (default), `tokens_to_mix` is a range; if True, `tokens_to_mix` is a list of indices
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    # trace_layers=None,  # List of traced outputs to return (not implemented in original code)
    return_first_pass_preds=False,      # If True, also return the prediction probs of first run (to reduce repetitive computations)
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)
    patch_spec_1st_pass = defaultdict(list)
    for t, l in states_to_patch_1st_pass:
        patch_spec_1st_pass[l].append(t)

    embed_layername = layername_uskg(model, "encoder", 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    
    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    if tokens_to_mix is None:
        tokens_to_mix_indices = None
    elif tokens_to_mix_individual_indices:
        tokens_to_mix_indices = tokens_to_mix
    else:
        tokens_to_mix_indices = list(range(*tokens_to_mix))

    if tokens_to_mix_1st_pass is None:
        tokens_to_mix_indices_1st_pass = None
    elif tokens_to_mix_individual_indices:
        tokens_to_mix_indices_1st_pass = tokens_to_mix_1st_pass
    else:
        tokens_to_mix_indices_1st_pass = list(range(*tokens_to_mix_1st_pass))

    # Define the model-patching rule for the 2nd (main) pass.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                # b, e = tokens_to_mix
                # x[1:, b:e] += noise * torch.from_numpy(
                #     prng(x.shape[0] - 1, e - b, x.shape[2])
                # ).to(x.device)
                # print(f'* 2nd pass, layer: {layer}, corrupting: {tokens_to_mix_indices}')
                mix_len = len(tokens_to_mix_indices)
                noise_data = noise_fn(
                    # torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                    torch.from_numpy(prng(x.shape[0] - 1, mix_len, x.shape[2]))
                ).to(device=x.device, dtype=x.dtype)

                if replace:
                    x[1:, tokens_to_mix_indices] = noise_data
                else:
                    x[1:, tokens_to_mix_indices] += noise_data
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        toks_to_patch = patch_spec.get(layer, [])
        toks_to_unpatch = unpatch_spec.get(layer, [])
        # if toks_to_patch:
        #     print(f'* 2nd pass, layer: {layer}, restoring: {toks_to_patch}')
        # if toks_to_unpatch:
        #     print(f'* 2nd pass, layer: {layer}, unpatching: {toks_to_unpatch}')

        for t in toks_to_patch:
            h[1:, t] = h[0, t]
        for t in toks_to_unpatch:
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # Define the model-patching rule for the 1st pass.
    def patch_rep_1st_pass(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix_1st_pass is not None:
                # b, e = tokens_to_mix
                # x[1:, b:e] += noise * torch.from_numpy(
                #     prng(x.shape[0] - 1, e - b, x.shape[2])
                # ).to(x.device)
                # print(f'* 1st pass, layer: {layer}, corrupting: {tokens_to_mix_indices_1st_pass}')
                mix_len = len(tokens_to_mix_indices_1st_pass)
                noise_data = noise_fn(
                    # torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                    torch.from_numpy(prng(x.shape[0] - 1, mix_len, x.shape[2]))
                ).to(device=x.device, dtype=x.dtype)

                if replace:
                    x[1:, tokens_to_mix_indices_1st_pass] = noise_data
                else:
                    x[1:, tokens_to_mix_indices_1st_pass] += noise_data
            return x
        # if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
        if layer not in patch_spec_1st_pass:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        toks_to_patch = patch_spec_1st_pass.get(layer, [])
        # if toks_to_patch:
        #     print(f'* 1st pass, layer: {layer}, restoring: {toks_to_patch}')
        for t in toks_to_patch:
            h[1:, t] = h[0, t]
        return x


    # With the patching rules defined, run the patched model in inference.
    first_pass_outputs_exp = None
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()) + list(patch_spec_1st_pass.keys()),
            edit_output=patch_rep_1st_pass if first_pass else patch_rep,
        ) as td:
            # outputs_exp = model(**inp)
            outputs_exp = run_model_forward_uskg(model, **inp)
            if first_pass:
                first_pass_trace = td
                first_pass_outputs_exp = outputs_exp

    # We report softmax probabilities for the answers_t token predictions of interest.
    # probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    # (YS) allowing multi-token answers_t
    all_ans_probs = []
    pass1_all_ans_probs = []
    try:
        _ = answers_t[0]
    except:
        # not sequence type, make it a list
        answers_t = [answers_t]

    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        probs = torch.softmax(outputs_exp.logits[1:, -len(answers_t) + i, :], dim=1).mean(dim=0)[_t]
        all_ans_probs.append(probs)

        if return_first_pass_preds:
            probs_1 = torch.softmax(outputs_exp.logits[1:, -len(answers_t) + i, :], dim=1).mean(dim=0)[_t]
            pass1_all_ans_probs.append(probs_1)

    if return_first_pass_preds:
        return all_ans_probs, pass1_all_ans_probs
    else:
        return all_ans_probs


def trace_with_repatch_uskg(*args, **kwargs):
    """ For compatibility, make this a wrapper for trace_with_patch_uskg_multi_token """

    ret = trace_with_repatch_uskg_multi_token(*args, **kwargs)
    if isinstance(ret, tuple):
        all_ans_probs, pass1_all_ans_probs = ret
        probs = min(all_ans_probs)
        pass1_probs = min(pass1_all_ans_probs)
        return probs, pass1_probs
    else:
        all_ans_probs = ret
        probs = min(all_ans_probs)
        return probs


def token_corruption_influence_uskg(
    mt,
    enc_sentence,
    dec_prompt,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    expect=None,
    device="cuda",
):
    """ Check corrupting each token does how much negative influence on prediction acc """
    inp = make_inputs_t5(
        mt.tokenizer,
        [enc_sentence] * (samples + 1),
        [dec_prompt] * (samples + 1),
        answer=expect,
        device=device
    )
    answer_len = 1
    if expect is not None:
        answer_len = len(mt.tokenizer.tokenize(expect))
    with torch.no_grad():
        answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]
    base_score = base_score.min().item()
    # [answer] = decode_tokens(mt.tokenizer, [answer_t])
    answer = decode_sentences(mt.tokenizer, answers_t)
    if expect is not None and answer.strip() != expect:
        return dict(
            expect=expect,
            clean_pred=answer,
            correct_prediction=False)

    input_tokens = decode_tokens(mt.tokenizer, inp['input_ids'][0])
    res = []

    # Single token
    for corrpt_idx in tqdm(range(len(inp['input_ids'][0]) - 1)):
        e_range = (corrpt_idx, corrpt_idx + 1)
        low_score = trace_with_patch_uskg(
            mt.model,
            inp=inp,
            states_to_patch=[], 
            answers_t=answers_t, 
            tokens_to_mix=e_range, 
            noise=noise, 
            uniform_noise=uniform_noise,
            replace=replace,
        ).item()

        res.append({
            'corrpt_idx': corrpt_idx,
            'corrpt_token': input_tokens[corrpt_idx],
            'corrpt_score': low_score,
            'corrpt_drop': base_score - low_score,
        })
    return res


def calculate_hidden_flow_uskg(
    mt,
    enc_sentence,
    dec_prompt,
    subject=None,
    e_range=None,
    samples=10,
    noise=0.1,
    # token_range=None,
    enc_token_range=None,
    dec_token_range=None,
    tokens_to_mix_individual_indices=False,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    sever_kind=None,
    expect=None,
):
    """
    AAA
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs_t5(mt.tokenizer, [enc_sentence] * (samples + 1), [dec_prompt] * (samples + 1), answer=expect)
    answer_len = 1
    if expect is not None:
        answer_len = len(mt.tokenizer.tokenize(expect))
    with torch.no_grad():
        answers_t, base_score = [d[0] for d in predict_from_input_uskg_multi_token(mt.model, inp, pred_len=answer_len)]
    base_score = min(base_score).item()
    answer = decode_sentences(mt.tokenizer, answers_t)
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False, is_good_sample=False)

    # e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    # text_range, struct_range = find_text_struct_in_range(mt.tokenizer, inp["input_ids"][0])
    if subject is not None:
        print('* Warning: calculate_hidden_flow_uskg(), "subject" arg not implemented')
    
    if e_range is None:
        tokens_to_mix = list(range(0, len(inp["input_ids"][0])))
    elif tokens_to_mix_individual_indices:
        tokens_to_mix = e_range
    else:
        tokens_to_mix = list(range(*e_range))
    # if token_range == "subject_last":
    #     token_range = [e_range[1] - 1]
    # if token_range is not None:
    #     raise ValueError(f"Unknown token_range: {token_range}")
    
    # (YS) for debugging
    _corrupted_input_toks = decode_tokens(mt.tokenizer, inp["input_ids"][0])
    for i in tokens_to_mix:
        _corrupted_input_toks[i] = '*' + _corrupted_input_toks[i]
    print('calculate_hidden_flow_uskg(): corrupted input:', ' '.join(_corrupted_input_toks))

    low_score = trace_with_patch_uskg(
        mt.model,
        inp=inp,
        states_to_patch=[], 
        answers_t=answers_t, 
        tokens_to_mix=tokens_to_mix, 
        noise=noise, 
        uniform_noise=uniform_noise,
        replace=replace,
        tokens_to_mix_individual_indices=True,
    ).item()

    result_dict = dict(
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0].detach().cpu().numpy().tolist(),
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        dec_input_ids=inp["decoder_input_ids"][0].detach().cpu().numpy().tolist(),
        dec_input_tokens=decode_tokens(mt.tokenizer, inp["decoder_input_ids"][0]),
        subject_range=tokens_to_mix,
        subject_range_individual_indices=True,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
        sever_kind=sever_kind or "",
    )

    if base_score - low_score < 0.5:
        result_dict['is_good_sample'] = False
        return result_dict

    if not kind:
        differences = trace_important_states_uskg(
            mt.model,
            mt.num_enc_layers,
            mt.num_dec_layers,
            inp,
            tokens_to_mix,
            answers_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            enc_token_range=enc_token_range,
            dec_token_range=dec_token_range,
            tokens_to_mix_individual_indices=True,
            sever_kind=sever_kind,
        )
    else:
        differences = trace_important_window_uskg(
            mt.model,
            mt.num_enc_layers,
            mt.num_dec_layers,
            inp,
            tokens_to_mix,
            answers_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            enc_token_range=enc_token_range,
            dec_token_range=dec_token_range,
            tokens_to_mix_individual_indices=True,
            sever_kind=sever_kind,
        )

    result_dict['scores'] = differences
    result_dict['is_good_sample'] = True
    return result_dict


def trace_important_states_uskg(
    model,
    num_enc_layers,
    num_dec_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    enc_token_range=None,
    dec_token_range=None,
    tokens_to_mix_individual_indices=False,
    sever_kind=None,
):
    """ 
    Args:
        e_range (List[int]): the tokens to corrupt (for now, it's only for encoder input. TODO: add another argument for decoder input corruption)
        enc|dec_token_range (List[int] or List[List[int]]): (updated) list of token ranges to try to restore (by default `None`, use all single tokens).
    """
    enc_ntoks = inp["input_ids"].shape[1]
    dec_ntoks = inp["decoder_input_ids"].shape[1]

    # table: (ntoks, nlayers)
    enc_table = []
    dec_table = []

    # (YS): Encoder part
    if enc_token_range is None:
        enc_token_range = range(enc_ntoks)
    if len(enc_token_range) > 0:
        enc_tqdm_pbar = tqdm(total=enc_ntoks*num_enc_layers,
                            desc="trace_important_states_uskg.encoder",
                            ascii=True)
        # enc_token_range = range(enc_ntoks)
        # if token_range is None:
        #     token_range = range(ntoks)
        for tnum in enc_token_range:
            ## YS: now tnum can be int or list
            ## TODO: finer control of states_to_unpatch, i.e. layers, part, etc.
            tnum_list = ensure_list(tnum)

            states_to_unpatch = []
            if sever_kind in ['self_attn', 'mlp']:
                states_to_unpatch = [(t, layername_uskg(model, 'encoder', l, sever_kind)) for l in range(num_enc_layers) for t in tnum_list]

            row = []
            for layer in range(num_enc_layers):
                r = trace_with_repatch_uskg(
                    model,
                    inp=inp,
                    states_to_patch=[(t, layername_uskg(model, 'encoder', layer)) for t in tnum_list],
                    states_to_unpatch=states_to_unpatch,
                    answers_t=answer_t,
                    tokens_to_mix=e_range,
                    tokens_to_mix_individual_indices=tokens_to_mix_individual_indices,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                )
                # YS: debug
                # r = torch.tensor(0, device="cuda:0")
                row.append(r)
                enc_tqdm_pbar.update(1)
            enc_table.append(torch.stack(row))
        enc_table_pt = torch.stack(enc_table).detach().cpu().numpy().tolist()
        enc_tqdm_pbar.close()
    else:
        # Do not analyse encoder restore
        enc_table_pt = []

    # (YS): Decoder part
    if dec_token_range is None:
        dec_token_range = range(dec_ntoks)
    if len(dec_token_range) > 0:
        dec_tqdm_pbar = tqdm(total=dec_ntoks*num_dec_layers,
                            desc="trace_important_states_uskg.decoder",
                            ascii=True)
        # dec_token_range = range(dec_ntoks)
        # if token_range is None:
        #     token_range = range(ntoks)
        for tnum in dec_token_range:
            tnum_list = ensure_list(tnum)

            states_to_unpatch = []
            if sever_kind in ['self_attn', 'cross_attn', 'mlp']:
                states_to_unpatch = [(t, layername_uskg(model, 'decoder', l, sever_kind)) for l in range(num_enc_layers) for t in tnum_list]

            row = []
            for layer in range(num_dec_layers):
                r = trace_with_repatch_uskg(
                    model,
                    inp=inp,
                    states_to_patch=[(t, layername_uskg(model, 'decoder', layer)) for t in tnum_list],
                    states_to_unpatch=states_to_unpatch,
                    answers_t=answer_t,
                    tokens_to_mix=e_range,
                    tokens_to_mix_individual_indices=tokens_to_mix_individual_indices,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                )
                row.append(r)
                dec_tqdm_pbar.update(1)
            dec_table.append(torch.stack(row))
        dec_table_pt = torch.stack(dec_table).detach().cpu().numpy().tolist()
        dec_tqdm_pbar.close()
    else:
        # Do not analyse decoder restore
        dec_table_pt = []

    return enc_table_pt, dec_table_pt


def trace_important_window_uskg(
    model,
    num_enc_layers,
    num_dec_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    enc_token_range=None,
    dec_token_range=None,
    tokens_to_mix_individual_indices=False,
):
    enc_ntoks = inp["input_ids"].shape[1]
    dec_ntoks = inp["decoder_input_ids"].shape[1]

    # table: (ntoks, nlayers)
    enc_table = []
    dec_table = []

    # YS TODO: add sever_kind

    # (YS): Encoder part
    if enc_token_range is None:
        enc_token_range = range(enc_ntoks)
    if len(enc_token_range) > 0:
        enc_tqdm_pbar = tqdm(total=enc_ntoks*num_enc_layers,
                            desc="trace_important_window_uskg.encoder",
                            ascii=True)
        for tnum in enc_token_range:
            row = []
            for layer in range(num_enc_layers):
                layerlist = [
                    (tnum, layername_uskg(model, 'encoder', L, kind))
                    for L in range(
                        max(0, layer - window // 2), min(num_enc_layers, layer - (-window // 2))
                    )
                ]
                r = trace_with_patch_uskg(
                    model,
                    inp,
                    layerlist,
                    answer_t,
                    tokens_to_mix=e_range,
                    tokens_to_mix_individual_indices=tokens_to_mix_individual_indices,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                )
                row.append(r)
                enc_tqdm_pbar.update(1)
            enc_table.append(torch.stack(row))
        enc_table_pt = torch.stack(enc_table).detach().cpu().numpy().tolist()
        enc_tqdm_pbar.close()
    else:
        enc_table_pt = []

    # (YS): Decoder part
    if dec_token_range is None:
        dec_token_range = range(dec_ntoks)
    if len(dec_token_range) > 0:
        dec_tqdm_pbar = tqdm(total=dec_ntoks*num_dec_layers,
                            desc="trace_important_window_uskg.decoder",
                            ascii=True)
        dec_token_range = range(dec_ntoks)
        for tnum in dec_token_range:
            row = []
            for layer in range(num_dec_layers):
                layerlist = [
                    (tnum, layername_uskg(model, 'decoder', L, kind))
                    for L in range(
                        max(0, layer - window // 2), min(num_dec_layers, layer - (-window // 2))
                    )
                ]
                r = trace_with_patch_uskg(
                    model,
                    inp,
                    layerlist,
                    answer_t,
                    tokens_to_mix=e_range,
                    tokens_to_mix_individual_indices=tokens_to_mix_individual_indices,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                )
                row.append(r)
                dec_tqdm_pbar.update(1)
            dec_table.append(torch.stack(row))
        dec_table_pt = torch.stack(dec_table).detach().cpu().numpy().tolist()
        dec_tqdm_pbar.close()
    else:
        dec_table_pt = []

    return enc_table_pt, dec_table_pt


def make_input_struct_corrupt():
    NotImplemented


def trace_struct_restore(
    mt,
    ex,
    subject_type='column',
    samples=10,
    noise=0.1,
    part='both',    # 'encoder', 'decoder', 'both'
    uniform_noise=False,
    replace=True,
    window=10,
    kind=None,
):
    """
    AAA
    Exp1
    """

    if part == 'encoder':
        enc_token_range = None
        dec_token_range = []
    elif part == 'decoder':
        enc_token_range = []
        dec_token_range = None
    elif part == 'both':
        enc_token_range = None
        dec_token_range = None
    else:
        raise ValueError(part)

    text_in = ex['text_in']
    struct_in = ex['struct_in']

    enc_sentence = f"{text_in}; structed knowledge: {struct_in}"
    enc_tokenized = mt.tokenizer(enc_sentence)

    token_ranges_dict = find_struct_name_ranges(mt.tokenizer, enc_tokenized['input_ids'], struct_in)

    if subject_type == 'column':
        node_name_ranges = token_ranges_dict['col_name_ranges']
    elif subject_type == 'table':
        node_name_ranges = token_ranges_dict['table_name_ranges']
    elif subject_type == 'value':
        node_name_ranges = token_ranges_dict['val_name_ranges']
    elif subject_type == 'db_id':
    #     token_name_ranges = token_ranges_dict['db_id_name_ranges']
        raise NotImplementedError('db_id is not used in sql')
    else:
        raise NotImplementedError(subject_type)

    sql_tokens = separate_punct(ex['seq_out']).split(' ')
    sql_nodes = set()
    for t in sql_tokens:
        if t in node_name_ranges:
            sql_nodes.add(t)

    all_results = []

    for node in sql_nodes:
        tok_ranges = node_name_ranges[node]
        tok_indices = [i for s, e in tok_ranges for i in range(s, e)]

        try:
            dec_prompt = make_dec_prompt(ex['seq_out'], node)
        except:
            breakpoint()

        result = calculate_hidden_flow_uskg(
            mt,
            enc_sentence=enc_sentence,
            dec_prompt=dec_prompt,
            expect=node,
            e_range=tok_indices,
            tokens_to_mix_individual_indices=True,
            samples=samples,
            noise=noise,
            # token_range=token_range,
            enc_token_range=enc_token_range,
            dec_token_range=dec_token_range,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
        )

        result['target_node'] = node      # actually already available in ['answer']
        result['db_id'] = ex['db_id']
        all_results.append(result)
    return all_results



def trace_section_corrupt_restore(
    mt,
    ex,
    subject_type='column',
    corrupt_section='text', # 'text', 'struct'
    samples=10,
    noise=0.1,
    part='encoder',         # For now, only 'encoder' supported for section corrupt
    uniform_noise=False,
    replace=True,
    window=10,
    kind=None,
):
    """
    AAA
    Exp 2.2
    """

    if part != 'encoder':
        raise ValueError(part)

    text_in = ex['text_in']
    struct_in = ex['struct_in']

    enc_sentence = f"{text_in}; structed knowledge: {struct_in}"
    enc_tokenized = mt.tokenizer(enc_sentence)

    text_range, struct_range = find_text_struct_in_range(mt.tokenizer, enc_tokenized['input_ids'])
    if corrupt_section == 'text':
        corrupt_tok_indices = list(range(*text_range))
    elif corrupt_section == 'struct':
        corrupt_tok_indices = list(range(*struct_range))
    else:
        raise ValueError(corrupt_section)

    token_ranges_dict = find_struct_name_ranges(mt.tokenizer, enc_tokenized['input_ids'], struct_in)

    if subject_type == 'column':
        node_name_ranges = token_ranges_dict['col_name_ranges']
    elif subject_type == 'table':
        node_name_ranges = token_ranges_dict['table_name_ranges']
    elif subject_type == 'value':
        node_name_ranges = token_ranges_dict['val_name_ranges']
    elif subject_type == 'db_id':
    #     token_name_ranges = token_ranges_dict['db_id_name_ranges']
        raise NotImplementedError('db_id is not used in sql')
    else:
        raise NotImplementedError(subject_type)

    sql_tokens = separate_punct(ex['seq_out']).split(' ')
    sql_nodes = set()
    for t in sql_tokens:
        if t in node_name_ranges:
            sql_nodes.add(t)

    all_results = []

    for node in sql_nodes:
        tok_ranges = node_name_ranges[node]
        enc_token_range = [[i for s, e in tok_ranges for i in range(s, e)]]
        dec_token_range = []

        try:
            dec_prompt = make_dec_prompt(ex['seq_out'], node)
        except:
            breakpoint()

        result = calculate_hidden_flow_uskg(
            mt,
            enc_sentence=enc_sentence,
            dec_prompt=dec_prompt,
            expect=node,
            e_range=corrupt_tok_indices,
            tokens_to_mix_individual_indices=True,
            samples=samples,
            noise=noise,
            # token_range=token_range,
            enc_token_range=enc_token_range,
            dec_token_range=dec_token_range,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
        )

        result['target_node'] = node      # actually already available in ['answer']
        result['db_id'] = ex['db_id']
        all_results.append(result)
    return all_results



class ModelAndTokenizer_USKG:
    """
    An object to hold on to (or automatically download and hold)
    a USKG model and tokenizer.  Counts the number of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        training_args=None,
        model_args=None,
        task_args=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        device="cuda",
    ):
        # if tokenizer is None:
        #     assert model_name is not None
        #     tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if model is None:
        #     assert model_name is not None
        #     model = AutoModelForSeq2SeqLM.from_pretrained(
        #         model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
        #     )
        #     nethook.set_requires_grad(False, model)
        #     model.eval().cuda()

        if (model is None) or (tokenizer is None):
            assert model_name is not None
            model, tokenizer, training_args, model_args, task_args = load_model_uskg(model_name, untie_embeddings=True)
        model = model.eval().to(device=device)

        self.tokenizer = tokenizer
        self.model = model
        self.training_args = training_args
        self.model_args = model_args
        self.task_args = task_args
        # Layer names in model:
        # encoder.embed_tokens
        # encoder.block.2
        # encoder.block.2.layer.0.SelfAttention
        # encoder.block.2.layer.1.DenseReluDense
        # decoder.embed_tokens
        # decoder.block.2
        # decoder.block.2.layer.0.SelfAttention
        # decoder.block.2.layer.1.EncDecAttention
        # decoder.block.2.layer.2.DenseReluDense
        self.enc_layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^pretrain_model\.encoder\.block\.\d+$", n))
        ]
        self.dec_layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^pretrain_model\.decoder\.block\.\d+$", n))
        ]
        self.layer_names = self.enc_layer_names + self.dec_layer_names

        self.num_enc_layers = len(self.enc_layer_names)
        self.num_dec_layers = len(self.dec_layer_names)
        self.num_layers = len(self.layer_names)
        

    def __repr__(self):
        return (
            f"ModelAndTokenizer_USKG(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername_uskg(model, part, num, kind=None):
    """
    part: encoder / decoder
    num: layer number
    kind: embed / self_attn / cross_attn / mlp / None
    """
    # if hasattr(model, "transformer"):
    #     if kind == "embed":
    #         return "transformer.wte"
    #     return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    # if hasattr(model, "gpt_neox"):
    #     if kind == "embed":
    #         return "gpt_neox.embed_in"
    #     if kind == "attn":
    #         kind = "attention"
    #     return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert hasattr(model.pretrain_model, part), f"{part} not in the model.pretrain_model of type {type(model.pretrain_model)}"

    # Layer names
    # decoder.block.2
    # decoder.block.2.layer.0.SelfAttention
    # decoder.block.2.layer.1.EncDecAttention
    # decoder.block.2.layer.2.DenseReluDense

    if kind == "embed":
        return f"pretrain_model.{part}.embed_tokens"

    _kind = None
    if kind == "self_attn":
        _kind = "layer.0.SelfAttention"
    elif kind == "cross_attn":
        assert part == "decoder", f"model {part} doesn't have {kind}"
        _kind = "layer.1.EncDecAttention"
    elif kind == "mlp":
        if part == "encoder":
            _kind = "layer.1.DenseReluDense"
        elif part == "decoder":
            _kind = "layer.2.DenseReluDense"
    
    return f"pretrain_model.{part}.block.{num}{'' if _kind is None else '.' + _kind}"


# def guess_subject(prompt):
#     return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
#         0
#     ].strip()


def plot_hidden_flow_uskg(
    mt,
    enc_sentence,
    dec_prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    """AAA"""
    if subject is None:
        # subject = guess_subject(prompt)
        # raise NotImplementedError('TODO: add automatic finding subject of different types (column, table, op, ...)')
        pass
    result = calculate_hidden_flow_uskg(
        mt,
        enc_sentence,
        dec_prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap_t5(result, savepdf)


def plot_trace_heatmap_t5(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    enc_differences, dec_differences = differences
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    if result.get("subject_range_individual_indices", False):
        subject_indices = result["subject_range"]
    else:
        subject_indices = list(range(*result["subject_range"]))
    # for i in range(*result["subject_range"]):
    for i in subject_indices:
        labels[i] = labels[i] + "*"
    dec_labels = list(result["dec_input_tokens"])

    def _draw_single_plot(fig, ax, differences=None, labels=None, modelname=modelname, title=title, part="all", kind=kind, answer=answer):
        if isinstance(differences, list):
            differences = numpy.array(differences)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels, fontsize=8)
        if not modelname:
            modelname = "USKG"
        if not kind:
            ax.set_title(f"Impact of restoring state after corrupted input\n")
            ax.set_xlabel(f"single restored layer within {modelname}.{part}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input\n")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        # cb = plt.colorbar(h)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # cb = fig.colorbar(h, cax=cax)
        cb = plt.colorbar(h, ax=ax)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)

    fig_height = (len(labels) + len(dec_labels) + 10) / 10
    # fig_h_ratio = (len(labels) + 5) / (len(dec_labels) + 5)
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, (ax0, ax1) = plt.subplots(
            nrows=2,
            figsize=(3.5, fig_height),
            dpi=200,
            gridspec_kw={
                'height_ratios': (len(labels) + 5, len(dec_labels) + 5)
            }
        )
        if len(enc_differences) > 0:
            _draw_single_plot(fig, ax0, enc_differences, labels, part="encoder")
        if len(dec_differences) > 0:
            _draw_single_plot(fig, ax1, dec_differences, dec_labels, part="decoder")
        fig.tight_layout()
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow_uskg(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow_uskg(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs_t5(tokenizer, enc_sentences, dec_prompts, answer=None, device="cuda"):
    enc_token_lists = [tokenizer.encode(p) for p in enc_sentences]
    enc_maxlen = max(len(t) for t in enc_token_lists)

    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0

    enc_input_ids = [[pad_id] * (enc_maxlen - len(t)) + t for t in enc_token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    enc_attention_mask = [[0] * (enc_maxlen - len(t)) + [1] * len(t) for t in enc_token_lists]
    enc_input_ids_tensor = torch.tensor(enc_input_ids).to(device)
    enc_attention_mask_tensor = torch.tensor(enc_attention_mask).to(device)

    if dec_prompts is not None:
        dec_token_lists = [[pad_id] + tokenizer.encode(p, add_special_tokens=False) for p in dec_prompts]
        if answer is not None:
            ans_token_lists = tokenizer.encode(answer, add_special_tokens=False)
            for t in dec_token_lists:
                t.extend(ans_token_lists[:-1])     # remove the last token, so the prediction last n tokens correspond to the answer
        dec_maxlen = max(len(t) for t in dec_token_lists)
        dec_input_ids = [[pad_id] * (dec_maxlen - len(t)) + t for t in dec_token_lists]
        dec_attention_mask = [[0] * (dec_maxlen - len(t)) + [1] * len(t) for t in dec_token_lists]
        dec_input_ids_tensor = torch.tensor(dec_input_ids).to(device)
        dec_attention_mask_tensor = torch.tensor(dec_attention_mask).to(device)
    else:
        dec_input_ids_tensor = None
        dec_attention_mask_tensor = None

    # return dict(
    #     input_ids=torch.tensor(enc_input_ids).to(device),
    #     #    position_ids=torch.tensor(position_ids).to(device),
    #     attention_mask=torch.tensor(enc_attention_mask).to(device),
    #     decoder_input_ids=torch.tensor(dec_input_ids).to(device),
    #     decoder_attention_mask=torch.tensor(dec_attention_mask).to(device),
    # )
    return dict(
        input_ids=enc_input_ids_tensor,
        attention_mask=enc_attention_mask_tensor,
        decoder_input_ids=dec_input_ids_tensor,
        decoder_attention_mask=dec_attention_mask_tensor,
    )


def predict_token_uskg(mt, enc_sentences, dec_prompts, return_p=False):
    inp = make_inputs_t5(mt.tokenizer, enc_sentences, dec_prompts)
    preds, p = predict_from_input_uskg(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input_uskg(model, inp):
    # out = model(**inp)
    out = run_model_forward_uskg(model, **inp)
    # print(out.keys())
    out = out["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def predict_from_input_uskg_multi_token(model, inp, pred_len=1):
    # out = model(**inp)
    out = run_model_forward_uskg(model, **inp)
    # print(out.keys())
    out = out["logits"]
    seq_len = out.size(1)
    probs = torch.softmax(out[:, seq_len - pred_len : seq_len], dim=-1)
    p, preds = torch.max(probs, dim=-1)
    return preds, p


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs_t5(mt.tokenizer, [s], [])
        with nethook.Trace(mt.model, layername_uskg(mt.model, "encoder", 0, "embed")) as t:
            run_model_forward_uskg(mt.model, **inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()
