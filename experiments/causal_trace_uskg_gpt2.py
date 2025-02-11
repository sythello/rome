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
    parse_sql_alias2table, nested_list_processing, nested_json_processing, \
    find_self_context_ranges

from util.uskg_gpt2 import ModelAndTokenizer_USKG_GPT2, run_model_forward_uskg_gpt2, layername_uskg_gpt2, \
    find_text_struct_in_range_gpt2, find_struct_name_ranges_gpt2, \
    USKG_GPT2_OUT_SPLITTER

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

from uskg.utils.dataset import gpt2_construct_input


def trace_with_patch_uskg_gpt2_multi_token(
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

    embed_layername = layername_uskg_gpt2(model, 0, "embed")

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
        outputs_exp = run_model_forward_uskg_gpt2(model, **inp)
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
    # try:
    #     _ = answers_t[0]
    # except:
    #     # not sequence type, make it a list
    #     answers_t = [answers_t]
    answers_t = ensure_list(answers_t)
    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        probs = torch.softmax(outputs_exp.logits[1:, -len(answers_t) + i, :], dim=-1).mean(dim=0)[_t]
        # probs = probs.detach().cpu().numpy().round(decimals=4).to_list()
        all_ans_probs.append(probs)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return all_ans_probs, all_traced

    return all_ans_probs


def trace_with_patch_uskg_gpt2(*args, **kwargs):
    """ For compatibility, make this a wrapper for trace_with_patch_uskg_multi_token """

    ret = trace_with_patch_uskg_gpt2_multi_token(*args, **kwargs)
    if isinstance(ret, tuple):
        all_ans_probs, all_traced = ret
        probs = min(all_ans_probs)
        return probs, all_traced
    else:
        all_ans_probs = ret
        probs = min(all_ans_probs)
        return probs




def run_repatch_uskg_gpt2_multi_token(
    model,  # The model
    inp,  # A set of inputs
    answer_len,         # Answer length to collect
    states_to_patch,    # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    states_to_corrupt=None,     # A list of (token index, layername) triples to corrupt; exclusive with `tokens_to_mix`
    tokens_to_mix=None,         # Range of tokens to corrupt (begin, end)
    # ------ 1st pass related args ------
    states_to_patch_1st_pass=None,      # states to restore for the 1st pass (default: empty) (Previously this is default to `list()` instead of None; not sure why is that)
    states_to_corrupt_1st_pass=None,    # A list of (token index, layername) triples to corrupt; exclusive with `tokens_to_mix_1st_pass`
    tokens_to_mix_1st_pass=None,        # tokens to corrupt in the 1st pass (default: None)
    # ------ other args ------
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
    if states_to_patch_1st_pass:
        for t, l in states_to_patch_1st_pass:
            patch_spec_1st_pass[l].append(t)

    embed_layername = layername_uskg_gpt2(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    
    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    # Decide the tokens to mix
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

    # Decide the corruption spec
    assert (states_to_corrupt is None) or (tokens_to_mix is None), \
        "Can't pass both `states_to_corrupt` and `tokens_to_mix`"
    corrupt_spec = defaultdict(list)
    if states_to_corrupt is not None:
        for t, l in states_to_corrupt:
            corrupt_spec[l].append(t)
    elif tokens_to_mix is not None:
        corrupt_spec[embed_layername] = tokens_to_mix_indices

    assert (states_to_corrupt_1st_pass is None) or (tokens_to_mix_1st_pass is None), \
        "Can't pass both `states_to_corrupt_1st_pass` and `tokens_to_mix_1st_pass`"
    corrupt_spec_1st_pass = defaultdict(list)
    if states_to_corrupt_1st_pass is not None:
        for t, l in states_to_corrupt_1st_pass:
            corrupt_spec_1st_pass[l].append(t)
    else:
        corrupt_spec_1st_pass[embed_layername] = tokens_to_mix_indices_1st_pass

    # Define the model-patching rule for the 2nd (main) pass.
    def patch_rep(x, layer):
        # if layer == embed_layername:
        #     # If requested, we corrupt a range of token embeddings on batch items x[1:]
        #     if tokens_to_mix is not None:
        #         mix_len = len(tokens_to_mix_indices)
        #         noise_data = noise_fn(
        #             torch.from_numpy(prng(x.shape[0] - 1, mix_len, x.shape[2]))
        #         ).to(device=x.device, dtype=x.dtype)

        #         if replace:
        #             x[1:, tokens_to_mix_indices] = noise_data
        #         else:
        #             x[1:, tokens_to_mix_indices] += noise_data
        #     return x

        # if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
        if (layer not in patch_spec) and (layer not in unpatch_spec) and (layer not in corrupt_spec):
            return x

        h = untuple(x)
        if layer in corrupt_spec:
            toks_to_mix = corrupt_spec[layer]
            if toks_to_mix:
                mix_len = len(toks_to_mix)
                noise_data = noise_fn(
                    torch.from_numpy(prng(h.shape[0] - 1, mix_len, h.shape[2]))
                ).to(device=h.device, dtype=h.dtype)

                if replace:
                    h[1:, toks_to_mix] = noise_data
                else:
                    h[1:, toks_to_mix] += noise_data

        # print(h.size())

        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
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

    # Define the model-patching rule for the 1st pass. (difference: no unpatch here)
    def patch_rep_1st_pass(x, layer):
        if (layer not in patch_spec_1st_pass) and (layer not in corrupt_spec_1st_pass):
            return x

        if layer in corrupt_spec_1st_pass:
            toks_to_mix = corrupt_spec_1st_pass[layer]
            if toks_to_mix:
                mix_len = len(toks_to_mix)
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, mix_len, x.shape[2]))
                ).to(device=x.device, dtype=x.dtype)

                if replace:
                    x[1:, toks_to_mix] = noise_data
                else:
                    x[1:, toks_to_mix] += noise_data

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
            list(corrupt_spec.keys()) + list(patch_spec.keys()) + list(unpatch_spec.keys()) + \
                list(corrupt_spec_1st_pass.keys()) + list(patch_spec_1st_pass.keys()),
            edit_output=patch_rep_1st_pass if first_pass else patch_rep,
        ) as td:
            # outputs_exp = model(**inp)
            outputs_exp = run_model_forward_uskg_gpt2(model, **inp)
            if first_pass:
                first_pass_trace = td
                first_pass_outputs_exp = outputs_exp

    # Here we report softmax probabilities for the answer positions.
    probs = torch.softmax(outputs_exp.logits[1:, -answer_len:, :], dim=-1).mean(dim=0)
    if return_first_pass_preds:
        probs_pass1 = torch.softmax(first_pass_outputs_exp.logits[1:, -answer_len:, :], dim=-1).mean(dim=0)

    if return_first_pass_preds:
        return probs, probs_pass1
    else:
        return probs



def trace_with_repatch_uskg_gpt2_multi_token(
    model,  # The model
    inp,  # A set of inputs
    answers_t,         # Answer length to collect
    states_to_patch,    # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    states_to_corrupt=None,     # A list of (token index, layername) triples to corrupt; exclusive with `tokens_to_mix`
    tokens_to_mix=None,         # Range of tokens to corrupt (begin, end)
    # ------ 1st pass related args ------
    states_to_patch_1st_pass=None,      # states to restore for the 1st pass (default: empty)
    states_to_corrupt_1st_pass=None,    # A list of (token index, layername) triples to corrupt; exclusive with `tokens_to_mix_1st_pass`
    tokens_to_mix_1st_pass=None,        # tokens to corrupt in the 1st pass (default: None)
    # ------ other args ------
    tokens_to_mix_individual_indices=False,     # If False (default), `tokens_to_mix` is a range; if True, `tokens_to_mix` is a list of indices
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    # trace_layers=None,  # List of traced outputs to return (not implemented in original code)
    return_first_pass_preds=False,      # If True, also return the prediction probs of first run (to reduce repetitive computations)
):
    answers_t = ensure_list(answers_t)

    ret = run_repatch_uskg_gpt2_multi_token(
        model=model,
        inp=inp,
        states_to_patch=states_to_patch,
        states_to_unpatch=states_to_unpatch,
        states_to_corrupt=states_to_corrupt,
        answer_len=len(answers_t),
        tokens_to_mix=tokens_to_mix,
        states_to_patch_1st_pass=states_to_patch_1st_pass,
        states_to_corrupt_1st_pass=states_to_corrupt_1st_pass,
        tokens_to_mix_1st_pass=tokens_to_mix_1st_pass,
        tokens_to_mix_individual_indices=tokens_to_mix_individual_indices,
        noise=noise,
        uniform_noise=uniform_noise,
        replace=replace,
        return_first_pass_preds=return_first_pass_preds,
    )

    if return_first_pass_preds:
        assert isinstance(ret, tuple), type(ret)
        vocab_probs, pass1_vocab_probs = ret
    else:
        vocab_probs = ret

    # We report softmax probabilities for the answers_t token predictions of interest.
    # probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    # (YS) allowing multi-token answers_t
    all_ans_probs = []
    pass1_all_ans_probs = []

    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        # vocab_probs: [answer_len, vocab_size]
        probs = vocab_probs[i, _t]
        all_ans_probs.append(probs)

        if return_first_pass_preds:
            probs_1 = pass1_vocab_probs[i, _t]
            pass1_all_ans_probs.append(probs_1)

    if return_first_pass_preds:
        return all_ans_probs, pass1_all_ans_probs
    else:
        return all_ans_probs


def trace_with_repatch_uskg_gpt2(*args, **kwargs):
    """ For compatibility, make this a wrapper for trace_with_patch_uskg_multi_token """

    ret = trace_with_repatch_uskg_gpt2_multi_token(*args, **kwargs)
    if isinstance(ret, tuple):
        all_ans_probs, pass1_all_ans_probs = ret
        probs = min(all_ans_probs)
        pass1_probs = min(pass1_all_ans_probs)
        return probs, pass1_probs
    else:
        all_ans_probs = ret
        probs = min(all_ans_probs)
        return probs



def run_attention_manip_uskg_gpt2_multi_token(
    model,  # The model
    inp,  # A set of inputs
    answer_len,         # Answer length to collect
    # layers_to_mix,      # A list of layername to corrupt. Has to be attention layers!
    # src_tokens_to_mix,    # In the attention mat, what src indices to mix.
    # tgt_tokens_to_mix,    # In the attention mat, what tgt indices to mix.
    mix_mask_per_layer=None,    # Dict[str, List|ndarray]: Full control of mixing, key = layer_name, value 1 = mix, 0 = keep 
    # noise=0.1,  # Level of noise to add
    # uniform_noise=False,
    attn_corrupt_type='weights',    # 'weights', 'logits'
    replace=True,  # True to replace with instead of add noise; TODO
):
    """
    AAA
    Tracing function specifically for manipulating attention weights / logits
    """

    all_hooks = []

    def _func_factory(mix_mask, attn_corrupt_type):
        def _attn_w_fn(attn):
            _mix_mask = mix_mask.to(device=attn.device)
            _zero = torch.tensor(0, dtype=attn.dtype, device=attn.device)
            attn[1:] = torch.where(_mix_mask, _zero, attn[1:])     # keep batch_idx=0 clean!
            return attn
        
        def _attn_lg_fn(attn):
            _mix_mask = mix_mask.to(device=attn.device)
            _neg = torch.tensor(-1e9, dtype=attn.dtype, device=attn.device)
            attn[1:] = torch.where(_mix_mask, _neg, attn[1:])     # keep batch_idx=0 clean!
            return attn

        def _pre_forward_hook_fn(m, inp):
            if attn_corrupt_type == 'weights':
                m.ext_attention_weights_fn = _attn_w_fn
            elif attn_corrupt_type == 'logits':
                m.ext_attention_logits_fn = _attn_lg_fn
            else:
                raise ValueError(attn_corrupt_type)

        def _forward_hook_fn(m, inp, outp):
            m.ext_attention_weights_fn = None
            m.ext_attention_logits_fn = None
    
        return _pre_forward_hook_fn, _forward_hook_fn


    for layer, mix_mask in mix_mask_per_layer.items():
        p_hook_fn, f_hook_fn = _func_factory(mix_mask, attn_corrupt_type)
        m = nethook.get_module(model, layer)
        # print(m)
        p_hook = m.register_forward_pre_hook(p_hook_fn)
        f_hook = m.register_forward_hook(f_hook_fn)
        all_hooks.extend([p_hook, f_hook])


    outputs_exp = run_model_forward_uskg(model, **inp)
    
    for hook in all_hooks:
        hook.remove()
    
    # Here we report softmax probabilities for the answer positions.
    probs = torch.softmax(outputs_exp.logits[1:, -answer_len:, :], dim=-1).mean(dim=0)
#     if return_first_pass_preds:
#         probs_pass1 = torch.softmax(first_pass_outputs_exp.logits[1:, -answer_len:, :], dim=-1).mean(dim=0)

#     if return_first_pass_preds:
#         return probs, probs_pass1
#     else:
#         return probs
    return probs

def trace_attention_manip_uskg_gpt2_multi_token(    
    model,  # The model
    inp,  # A set of inputs
    answers_t,         # Answer tensor
    # --- Deprecated ---
    layers_to_mix=None,      # A list of layername to corrupt. Has to be attention layers!
    src_tokens_to_mix=None,    # In the attention mat, what src indices to mix.
    tgt_tokens_to_mix=None,    # In the attention mat, what tgt indices to mix.
    # --- END Deprecated ---
    mix_mask_per_layer=None,    # Dict[str, List|ndarray]: Full control of mixing, key = layer_name, value 1 = mix, 0 = keep 
    # noise=0.1,  # Level of noise to add
    # uniform_noise=False,
    replace=True,  # True to replace with instead of add noise
    attn_corrupt_type='weights',    # 'weights', 'logits'
    # trace_layers=None,  # List of traced outputs to return (not implemented in original code)
    # return_first_pass_preds=False,      # If True, also return the prediction probs of first run (to reduce repetitive computations)
):

    if mix_mask_per_layer is None:
        assert (None not in (layers_to_mix, src_tokens_to_mix, tgt_tokens_to_mix))

        if not trace_attention_manip_uskg_gpt2_multi_token._warned_deprecate:
            print('*** trace_attention_manip_uskg_multi_token():')
            print('*** Deprecated usage: layers_to_mix, src_tokens_to_mix, tgt_tokens_to_mix')
            trace_attention_manip_uskg_gpt2_multi_token._warned_deprecate = True

        bs, seq_len = inp['input_ids'].size()
        prev_len = model.preseqlen

        _mask = torch.zeros(1, 1, seq_len, seq_len + prev_len).bool()
        _t = _mask[:, :, src_tokens_to_mix]
        _t[:, :, :, tgt_tokens_to_mix] = True
        _mask[:, :, src_tokens_to_mix] = _t

        mix_mask_per_layer = {l : _mask for l in layers_to_mix}

    vocab_probs = run_attention_manip_uskg_gpt2_multi_token(
        model=model,
        inp=inp,
        answer_len=len(answers_t),
        # layers_to_mix=layers_to_mix,
        # src_tokens_to_mix=src_tokens_to_mix,
        # tgt_tokens_to_mix=tgt_tokens_to_mix,
        mix_mask_per_layer=mix_mask_per_layer,
        replace=replace,
        attn_corrupt_type=attn_corrupt_type,
    )

    # We report softmax probabilities for the answers_t token predictions of interest.
    # probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    # (YS) allowing multi-token answers_t
    all_ans_probs = []
    # pass1_all_ans_probs = []

    for i, _t in enumerate(answers_t):
        # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
        # vocab_probs: [answer_len, vocab_size]
        probs = vocab_probs[i, _t]
        all_ans_probs.append(probs)

    #     if return_first_pass_preds:
    #         probs_1 = pass1_vocab_probs[i, _t]
    #         pass1_all_ans_probs.append(probs_1)

    # if return_first_pass_preds:
    #     return all_ans_probs, pass1_all_ans_probs
    # else:
    #     return all_ans_probs

    probs = min(all_ans_probs)
    return probs

trace_attention_manip_uskg_gpt2_multi_token._warned_deprecate = False



# def run_layer_copy_uskg_multi_token(
#     model,  # The model
#     inp,  # A set of inputs
#     answer_len,         # Answer length to collect
#     # TODO perhaps: to enable patch and unpatch
#     # states_to_patch,    # A list of (token index, layername) triples to restore (set to clean state)
#     # states_to_unpatch,  # A list of (token index, layername) triples to re-randomize (set to 1st run state)
#     states_to_copy_from,
#     states_to_copy_to,
#     states_to_corrupt=None,     # A list of (token index, layername) triples to corrupt (set to random state)
#     # ------ 1st pass related args ------
#     # states_to_patch_1st_pass=None,      # A list of (token index, layername) triples to restore in the 1st pass
#     states_to_corrupt_1st_pass=None,    # A list of (token index, layername) triples to corrupt in the 1st pass
#     # ------ other args ------
#     noise=0.1,  # Level of noise to add
#     uniform_noise=False,
#     replace=False,  # True to replace with instead of add noise
#     # trace_layers=None,  # List of traced outputs to return (not implemented in original code)
#     return_first_pass_preds=False,      # If True, also return the prediction probs of first run (to reduce repetitive computations)
# ):
    
#     """
#     AAA
#     Layer copy: put the intermediate values of a (lower) layer into another (higher) layer, effectively skipping certain layers
#     (Not used for now; for exp7, directly using sublayer-zero is conceptually faster since it's single-run)
#     """

#     rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
#     if uniform_noise:
#         prng = lambda *shape: rs.uniform(-1, 1, shape)
#     else:
#         prng = lambda *shape: rs.randn(*shape)
#     copy_from_spec = defaultdict(list)
#     for t, l in states_to_copy_from:
#         copy_from_spec[l].append(t)
#     copy_to_spec = defaultdict(list)
#     for t, l in states_to_copy_to:
#         copy_to_spec[l].append(t)
#     # patch_spec_1st_pass = defaultdict(list)
#     # if states_to_patch_1st_pass:
#     #     for t, l in states_to_patch_1st_pass:
#     #         patch_spec_1st_pass[l].append(t)

#     assert len(states_to_copy_to) == len(states_to_copy_from), (len(states_to_copy_to), len(states_to_copy_from))
#     # dest_layer -> src_layer
#     layer_copy_dict = dict(zip(states_to_copy_to, states_to_copy_from))

#     embed_layername = layername_uskg(model, "encoder", 0, "embed")

#     def untuple(x):
#         return x[0] if isinstance(x, tuple) else x
    
#     # Define the model-patching rule.
#     if isinstance(noise, float):
#         noise_fn = lambda x: noise * x
#     else:
#         noise_fn = noise

#     # Decide the corruption spec
#     corrupt_spec = defaultdict(list)
#     if states_to_corrupt is not None:
#         for t, l in states_to_corrupt:
#             corrupt_spec[l].append(t)
#     corrupt_spec_1st_pass = defaultdict(list)
#     if states_to_corrupt_1st_pass is not None:
#         for t, l in states_to_corrupt_1st_pass:
#             corrupt_spec_1st_pass[l].append(t)


#     # Define the model-patching rule for the 2nd (main) pass.
#     def patch_rep(x, layer):
#         if (layer not in copy_to_spec) and (layer not in corrupt_spec):
#             return x

#         h = untuple(x)
#         if layer in corrupt_spec:
#             toks_to_mix = corrupt_spec[layer]
#             if toks_to_mix:
#                 mix_len = len(toks_to_mix)
#                 noise_data = noise_fn(
#                     torch.from_numpy(prng(h.shape[0] - 1, mix_len, h.shape[2]))
#                 ).to(device=h.device, dtype=h.dtype)

#                 if replace:
#                     h[1:, toks_to_mix] = noise_data
#                 else:
#                     h[1:, toks_to_mix] += noise_data

#         # If this layer is in the patch_spec, restore the uncorrupted hidden state
#         # for selected tokens.
#         # toks_to_patch = patch_spec.get(layer, [])
#         # toks_to_unpatch = unpatch_spec.get(layer, [])
#         toks_to_copy_to = copy_to_spec.get(layer, [])
#         # if toks_to_patch:
#         #     print(f'* 2nd pass, layer: {layer}, restoring: {toks_to_patch}')
#         # if toks_to_unpatch:
#         #     print(f'* 2nd pass, layer: {layer}, unpatching: {toks_to_unpatch}')

#         # for t in toks_to_patch:
#         #     h[1:, t] = h[0, t]
#         # for t in toks_to_unpatch:
#         #     h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
#         for t in toks_to_copy_to:
#             src_t, src_layer = layer_copy_dict[(t, layer)]
#             h[1:, t] = untuple(first_pass_trace[src_layer].output)[1:, src_t]
        
#         return x

#     # Define the model-patching rule for the 1st pass. (difference: no unpatch here)
#     def patch_rep_1st_pass(x, layer):
#         if (layer not in corrupt_spec_1st_pass):
#             return x

#         if layer in corrupt_spec_1st_pass:
#             toks_to_mix = corrupt_spec_1st_pass[layer]
#             if toks_to_mix:
#                 mix_len = len(toks_to_mix)
#                 noise_data = noise_fn(
#                     torch.from_numpy(prng(x.shape[0] - 1, mix_len, x.shape[2]))
#                 ).to(device=x.device, dtype=x.dtype)

#                 if replace:
#                     x[1:, toks_to_mix] = noise_data
#                 else:
#                     x[1:, toks_to_mix] += noise_data

#         # If this layer is in the patch_spec, restore the uncorrupted hidden state
#         # for selected tokens.
#         # h = untuple(x)
#         # toks_to_patch = patch_spec_1st_pass.get(layer, [])
#         # if toks_to_patch:
#         #     print(f'* 1st pass, layer: {layer}, restoring: {toks_to_patch}')

#         # for t in toks_to_patch:
#         #     h[1:, t] = h[0, t]
#         return x


#     # With the patching rules defined, run the patched model in inference.
#     first_pass_outputs_exp = None
#     for first_pass in [True, False]:
#         with torch.no_grad(), nethook.TraceDict(
#             model,
#             list(corrupt_spec.keys()) + list(copy_from_spec.keys()) + list(copy_to_spec.keys()) + \
#                 list(corrupt_spec_1st_pass.keys()),
#             edit_output=patch_rep_1st_pass if first_pass else patch_rep,
#         ) as td:
#             # outputs_exp = model(**inp)
#             outputs_exp = run_model_forward_uskg(model, **inp)
#             if first_pass:
#                 first_pass_trace = td
#                 first_pass_outputs_exp = outputs_exp

#     # Here we report softmax probabilities for the answer positions.
#     probs = torch.softmax(outputs_exp.logits[1:, -answer_len:, :], dim=-1).mean(dim=0)
#     if return_first_pass_preds:
#         probs_pass1 = torch.softmax(first_pass_outputs_exp.logits[1:, -answer_len:, :], dim=-1).mean(dim=0)

#     if return_first_pass_preds:
#         return probs, probs_pass1
#     else:
#         return probs

# def trace_layer_copy_uskg_multi_token(**kwargs):
#     ## NOTE: untested!

#     run_kwargs = dict(kwargs)
#     run_kwargs['answer_len'] = len(kwargs['answers_t'])
#     del run_kwargs['answers_t']

#     if kwargs['return_first_pass_preds']:
#         vocab_probs, pass1_vocab_probs = run_attention_manip_uskg_multi_token(**run_kwargs)
#     else:
#         vocab_probs = run_attention_manip_uskg_multi_token(**run_kwargs)

#     all_ans_probs = []
#     # pass1_all_ans_probs = []

#     for i, _t in enumerate(kwargs['answers_t']):
#         # let len(answers_t) = n, then the positions of interest are [-n, -n+1, ..., -1]
#         # vocab_probs: [answer_len, vocab_size]
#         probs = vocab_probs[i, _t]
#         all_ans_probs.append(probs)

#         # if kwargs['return_first_pass_preds']:
#         #     probs_1 = pass1_vocab_probs[i, _t]
#         #     pass1_all_ans_probs.append(probs_1)

#     # if return_first_pass_preds:
#     #     return all_ans_probs, pass1_all_ans_probs
#     # else:
#     #     return all_ans_probs
#     probs = min(all_ans_probs)
#     return probs



def build_self_attention_mask_gpt2(
    a_ex,
    seq_len=None,
    mt=None,
    prefix_len=10,      # available in mt.model.preseqlen (but maybe mt is not provided)
    use_self_node=False):
    """
    BBB
    For attention-related experiments: build self-attention masks across sections

    Args:
    a_ex (Dict)
    seq_len (int): sequence length
    prefix_len (int): prefix length
    use_self_node (bool): whether to include sections for self_node and struct_context
    """

    if prefix_len is None:
        prefix_len = mt.model.preseqlen

    enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']
    answer_len = len(a_ex['answers_t'])

    text_range = a_ex['text_range']
    struct_range = a_ex['struct_range']
    text_st, text_ed = text_range
    struct_st, struct_ed = struct_range
    # text_tok_indices = list(range(*text_range))
    # struct_tok_indices = list(range(*struct_range))

    if use_self_node:
        expect_input_ranges = a_ex['expect_input_ranges']
        tok_indices = [i for s, e in expect_input_ranges for i in range(s, e)]

        self_ranges = a_ex['self_ranges']
        struct_context_ranges = a_ex['context_ranges']

        self_tok_indices = [i for s, e in self_ranges for i in range(s, e)]
        struct_context_tok_indices = [i for s, e in struct_context_ranges for i in range(s, e)]

        self_tok_indices_tgt_side = [i + prefix_len for i in self_tok_indices]
        struct_context_tok_indices_tgt_side = [i + prefix_len for i in struct_context_tok_indices]

    if seq_len is None:
        # need to tokenize and decide seq_len
        # TODO: untested!
        assert mt is not None
        _tok_ids = mt.tokenizer.encode(enc_sentence)
        seq_len = len(_tok_ids)


    att_mix_mask_dict = dict()
    # mix_mask: (batch, head, src_len, tgt_len)
    # NOTE: assumes no left padding here. If has padding, need to move the indices

    # In current format, [prefix, text, struct, out], so text->struct doesn't exist

    # t2s_mask = torch.zeros(1, 1, seq_len, seq_len + prefix_len).bool()
    # t2s_mask[:, :, text_st : text_ed, struct_st + prefix_len : struct_ed + prefix_len] = True
    # att_mix_mask_dict['t->s'] = t2s_mask

    all_mask = torch.ones(1, 1, seq_len, seq_len + prefix_len).bool()
    att_mix_mask_dict['all'] = all_mask

    s2t_mask = torch.zeros_like(all_mask).bool()
    s2t_mask[:, :, struct_st : struct_ed, text_st + prefix_len : text_ed + prefix_len] = True
    att_mix_mask_dict['s->t'] = s2t_mask

    # att_mix_mask_dict['t<->s'] = t2s_mask | s2t_mask

    t2p_mask = torch.zeros_like(all_mask).bool()
    t2p_mask[:, :, text_st : text_ed, :prefix_len] = True
    att_mix_mask_dict['t->p'] = t2p_mask

    s2p_mask = torch.zeros_like(all_mask).bool()
    s2p_mask[:, :, struct_st : struct_ed, :prefix_len] = True
    att_mix_mask_dict['s->p'] = s2p_mask

    att_mix_mask_dict['ts->p'] = t2p_mask | s2p_mask

    # ADDED: section self-attention
    t2t_mask = torch.zeros_like(all_mask).bool()
    t2t_mask[:, :, text_st : text_ed, text_st + prefix_len : text_ed + prefix_len] = True
    att_mix_mask_dict['t->t'] = t2t_mask

    s2s_mask = torch.zeros_like(all_mask).bool()
    s2s_mask[:, :, struct_st : struct_ed, struct_st + prefix_len : struct_ed + prefix_len] = True
    att_mix_mask_dict['s->s'] = s2s_mask


    ans2t_mask = torch.zeros_like(all_mask)
    ans2t_mask[:, :, -answer_len:, text_st + prefix_len : text_ed + prefix_len] = True
    att_mix_mask_dict['ans->t'] = ans2t_mask
    all2t_mask = torch.zeros_like(all_mask)
    all2t_mask[:, :, :, text_st + prefix_len : text_ed + prefix_len] = True
    att_mix_mask_dict['all->t'] = all2t_mask

    ans2s_mask = torch.zeros_like(all_mask)
    ans2s_mask[:, :, -answer_len:, struct_st + prefix_len : struct_ed + prefix_len] = True
    att_mix_mask_dict['ans->s'] = ans2s_mask
    all2s_mask = torch.zeros_like(all_mask)
    all2s_mask[:, :, :, struct_st + prefix_len : struct_ed + prefix_len] = True
    att_mix_mask_dict['all->s'] = all2s_mask

    ans2p_mask = torch.zeros_like(all_mask)
    ans2p_mask[:, :, -answer_len:, :prefix_len] = True
    att_mix_mask_dict['ans->p'] = ans2p_mask
    all2p_mask = torch.zeros_like(all_mask)
    all2p_mask[:, :, :, :prefix_len] = True
    att_mix_mask_dict['all->p'] = all2p_mask

    # NEW: other toks (o), include the connector and final </s>
    ans2o_mask = torch.zeros_like(all_mask)
    ans2o_mask[:, :, -answer_len:, text_ed + prefix_len : struct_st + prefix_len] = True
    ans2o_mask[:, :, -answer_len:, struct_ed + prefix_len:] = True
    att_mix_mask_dict['ans->o'] = ans2o_mask
    all2o_mask = torch.zeros_like(all_mask)
    all2o_mask[:, :, :, text_ed + prefix_len : struct_st + prefix_len] = True
    all2o_mask[:, :, :, struct_ed + prefix_len:] = True
    att_mix_mask_dict['all->o'] = all2o_mask

    att_mix_mask_dict['ans->t+o'] = ans2t_mask | ans2o_mask
    att_mix_mask_dict['all->t+o'] = all2t_mask | all2o_mask

    if use_self_node:
        # ADDED: regarding struct context
        # Notice that it's ok to have 1 list in indexing, but not ok to have 2
        # If there are 2 lists, it will become a "gather()" which treats the 2 lists in a zipped way
        s2c_mask = torch.zeros_like(all_mask).bool()
        s2c_mask[:, :, struct_st : struct_ed, struct_context_tok_indices_tgt_side] = True
        att_mix_mask_dict['s->c'] = s2c_mask

        c2p_mask = torch.zeros_like(all_mask).bool()
        c2p_mask[:, :, struct_context_tok_indices, :prefix_len] = True
        att_mix_mask_dict['c->p'] = c2p_mask

        # c2t: skipped, as already see even s2t is not so effective

        c2s_mask = torch.zeros_like(all_mask).bool()
        c2s_mask[:, :, struct_context_tok_indices, struct_st + prefix_len : struct_ed + prefix_len] = True
        att_mix_mask_dict['c->s'] = c2s_mask

        c2c_mask = c2s_mask.clone()
        c2c_mask[:, :, :, self_tok_indices_tgt_side] = False
        assert c2c_mask.sum().item() == len(struct_context_tok_indices) ** 2, \
            (c2c_mask.sum().item(), len(struct_context_tok_indices) ** 2)
        att_mix_mask_dict['c->c'] = c2c_mask


        ans2c_mask = torch.zeros_like(all_mask).bool()
        ans2c_mask[:, :, -answer_len:, struct_context_tok_indices_tgt_side] = True
        att_mix_mask_dict['ans->c'] = ans2c_mask
        all2c_mask = torch.zeros_like(all_mask).bool()
        all2c_mask[:, :, :, struct_context_tok_indices_tgt_side] = True
        att_mix_mask_dict['all->c'] = all2c_mask

        ans2self_mask = torch.zeros_like(all_mask).bool()
        ans2self_mask[:, :, -answer_len:, self_tok_indices_tgt_side] = True
        att_mix_mask_dict['ans->self'] = ans2self_mask
        all2self_mask = torch.zeros_like(all_mask).bool()
        all2self_mask[:, :, :, self_tok_indices_tgt_side] = True
        att_mix_mask_dict['all->self'] = all2self_mask

    return att_mix_mask_dict



def calculate_hidden_flow_uskg_gpt2(
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
    inp = make_inputs_gpt2(mt.tokenizer, [enc_sentence] * (samples + 1), [dec_prompt] * (samples + 1), answer=expect)
    answer_len = 1
    if expect is not None:
        answer_len = len(mt.tokenizer.tokenize(expect))
    with torch.no_grad():
        answers_t, base_score = [d[0] for d in predict_from_input_uskg_gpt2_multi_token(mt.model, inp, pred_len=answer_len)]
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

    low_score = trace_with_patch_uskg_gpt2(
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
        differences = trace_important_states_uskg_gpt2(
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
        differences = trace_important_window_uskg_gpt2(
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


def trace_important_states_uskg_gpt2(
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
        enc_tqdm_pbar = tqdm(total=len(enc_token_range)*num_enc_layers,
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
                r = trace_with_repatch_uskg_gpt2(
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
        dec_tqdm_pbar = tqdm(total=len(dec_token_range)*num_dec_layers,
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
                r = trace_with_repatch_uskg_gpt2(
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


def trace_important_window_uskg_gpt2(
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

    # YS TODO:
    # add sever_kind
    # support enc_token_range: List[List[int]]

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
                r = trace_with_patch_uskg_gpt2(
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
                r = trace_with_patch_uskg_gpt2(
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



def plot_hidden_flow_uskg_gpt2(
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
    result = calculate_hidden_flow_uskg_gpt2(
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
    plot_trace_heatmap_gpt2(result, savepdf)


def plot_trace_heatmap_gpt2(result, savepdf=None, title=None, xlabel=None, modelname=None):
    ## YS NOTE: copied from original (causal_trace.py)
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
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
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()



# def plot_all_flow_uskg(mt, prompt, subject=None):
#     for kind in ["mlp", "attn", None]:
#         plot_hidden_flow_uskg(mt, prompt, subject, kind=kind)



# Utilities for dealing with tokens
def make_inputs_gpt2(tokenizer, prompts, answer=None, device="cuda"):
    # Copied (and modified) from original ROME code for gpt2 (causal_trace.py)
    token_lists = [tokenizer.encode(p) for p in prompts]

    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    
    if answer is not None:
        ans_token_lists = tokenizer.encode(answer, add_special_tokens=False)
        for t in token_lists:
            t.extend(ans_token_lists[:-1])     # remove the last token, so the prediction last n tokens correspond to the answer

    maxlen = max(len(t) for t in token_lists)

    if any([len(t) != maxlen for t in token_lists]):
        print('** make_inputs_gpt2(): input lengths different!')
        print(f'** {[len(t) for t in token_lists]}')

    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def predict_from_input_uskg_gpt2_multi_token(model, inp, pred_len=1):
    # out = model(**inp)
    with torch.no_grad():
        out = run_model_forward_uskg_gpt2(model, **inp)
    # print(out.keys())
    out = out["logits"]
    seq_len = out.size(1)
    # out: (bsz, seq_len, vocab_size)
    probs = torch.softmax(out[:, seq_len - pred_len : seq_len], dim=-1)
    p, preds = torch.max(probs, dim=-1)
    return preds, p

def predict_with_repatch_gpt2(
        model, 
        inp, 
        pred_len,
        states_to_patch,    # A list of (token index, layername) triples to restore
        states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
        states_to_corrupt=None,
        tokens_to_mix=None,  # Range of tokens to corrupt (begin, end)
        states_to_patch_1st_pass=None,      # states to restore for the 1st pass (default: empty)
        states_to_corrupt_1st_pass=None,
        tokens_to_mix_1st_pass=None,        # tokens to corrupt in the 1st pass (default: None)
        tokens_to_mix_individual_indices=False,     # If False (default), `tokens_to_mix` is a range; if True, `tokens_to_mix` is a list of indices
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        # trace_layers=None,  # List of traced outputs to return (not implemented in original code)
        # return_first_pass_preds=False,      # If True, also return the prediction probs of first run (to reduce repetitive computations)
    ):
    vocab_probs = run_repatch_uskg_gpt2_multi_token(
        model=model,
        inp=inp,
        states_to_patch=states_to_patch,
        states_to_unpatch=states_to_unpatch,
        states_to_corrupt=states_to_corrupt,
        answer_len=pred_len,
        tokens_to_mix=tokens_to_mix,
        states_to_patch_1st_pass=states_to_patch_1st_pass,
        states_to_corrupt_1st_pass=states_to_corrupt_1st_pass,
        tokens_to_mix_1st_pass=tokens_to_mix_1st_pass,
        tokens_to_mix_individual_indices=tokens_to_mix_individual_indices,
        noise=noise,
        uniform_noise=uniform_noise,
        replace=replace,
        return_first_pass_preds=False,
    )

    # trace_with_repatch_uskg_multi_token()

    # vocab_probs: (answer_len, vocab_size)
    p, preds = torch.max(vocab_probs, dim=-1)
    return preds, p


def add_basic_analysis_info_gpt2(
        mt,
        ex):
    """
    BBBB
    Add basic information for ex (dict) that is needed by all kinds of analysis
    """

    text_in = ex['text_in']
    struct_in = ex['struct_in']

    enc_sentence = f"{text_in}; structed knowledge: {struct_in}"
    tokenized_item = gpt2_construct_input(
        seq_in=enc_sentence,
        seq_out=ex['seq_out'],
        tokenizer=mt.tokenizer,
        in_maxlen=362,
        out_maxlen=490,
        padding=False,
    )

    text_range, struct_range, sql_range = find_text_struct_in_range_gpt2(mt.tokenizer, tokenized_item['input_ids'])

    # enc_sentence = f"{text_in}; structed knowledge: {struct_in}"
    # enc_tokenized = mt.tokenizer(enc_sentence)
    ex['enc_sentence'] = enc_sentence
    ex['tokenized_item'] = tokenized_item

    # All tokens before SQL, should end with "; SQL:"
    # Does not work, because this (tokenized again) differs from original
    # ex['pre_sql_sequence'] = mt.tokenizer.decode(tokenized_item['input_ids'][:sql_range[0]])

    # Assuming enc_sentence is not truncated. For analysis, can skip those need truncation
    _sql_connector = USKG_GPT2_OUT_SPLITTER
    ex['pre_sql_sequence'] = enc_sentence + ' ' + _sql_connector        

    # text_range, struct_range = find_text_struct_in_range(mt.tokenizer, enc_tokenized['input_ids'])
    ex['text_range'] = text_range
    ex['struct_range'] = struct_range
    ex['sql_range'] = sql_range
    
    parsed_struct_in = parse_struct_in(struct_in)
    ex['parsed_struct_in'] = parsed_struct_in

    alias2table = parse_sql_alias2table(ex['seq_out'])
    ex['alias2table'] = alias2table

    col2table = defaultdict(list)
    ## Counters for struct_in
    col_name_counter = Counter()
    tab_name_counter = Counter()
    # node_name_counter = Counter()
    db_id_t, tables = parsed_struct_in
    for table_name_t, cols in tables:
        _, table_name, _ = table_name_t
        tab_name_counter[table_name] += 1
        for col_name_t, vals in cols:
            _, col_name, _ = col_name_t
            col2table[col_name].append(table_name)
            col_name_counter[col_name] += 1
    
    ex['col2table'] = col2table
    ex['col_name_counter'] = col_name_counter
    ex['tab_name_counter'] = tab_name_counter

    # YS NOTE: Merging col and tab to remove "col & table" nodes when remove_dup = True,
    # since currently make_dec_prompt() can't distinguish node type from SQL 
    # node_name_counter = col_name_counter + tab_name_counter

    # find_struct_name_ranges(): needs "struct_in", "enc_sentence"; adds "struct_node_ranges_dict"
    token_ranges_dict = find_struct_name_ranges_gpt2(mt.tokenizer, ex)
    # ex['struct_node_ranges_dict'] = token_ranges_dict   --> already added

    # sql_tokens = separate_punct(ex['seq_out']).strip().lower().split(' ')
    sql_token_ranges = separate_punct_by_offset(ex['seq_out'])
    sql_tokens = [ex['seq_out'][s:e] for s, e in sql_token_ranges]
    tok_ranges2type = categorize_tokens_offset(ex['seq_out'], sql_token_ranges)

    type2tok_ranges = defaultdict(list)
    for _rg, _type in tok_ranges2type.items():
        type2tok_ranges[_type].append(_rg)

    ex['sql_tokens'] = sql_tokens
    ex['sql_token_ranges'] = sql_token_ranges
    ex['tok_ranges2type'] = tok_ranges2type
    ex['type2tok_ranges'] = type2tok_ranges

    sql_col_nodes = set()
    sql_tab_nodes = set()
    sql_alias_nodes = set()
    for _rg, _type in tok_ranges2type.items():
        s, e = _rg
        _tok = ex['seq_out'][s:e]
        if _type == 'column':
            sql_col_nodes.add(_tok)
        elif _type == 'table':
            sql_tab_nodes.add(_tok)
        elif _type == 'table_alias':
            sql_alias_nodes.add(_tok)

    ## Sanity check: SQL node should occur in DB struct_in
    for t in sql_col_nodes:
        if col_name_counter[t] == 0:
            err_msg = f'Column {t} not found in struct_in: {struct_in}! (SQL: {ex["seq_out"]})'
            raise SDRASampleError(err_msg)
    for t in sql_tab_nodes:
        if tab_name_counter[t] == 0:
            err_msg = f'Table {t} not found in struct_in: {struct_in}! (SQL: {ex["seq_out"]})'
            raise SDRASampleError(err_msg)

    ex['sql_col_nodes'] = sql_col_nodes
    ex['sql_tab_nodes'] = sql_tab_nodes
    ex['sql_alias_nodes'] = sql_alias_nodes



def create_analysis_sample_dicts_gpt2(
        mt, 
        ex, 
        subject_type,
        remove_struct_duplicate_nodes=True):
    """
    BBB
    Create new sample dicts for analysis purpose 
    Return:
        analysis_ex_dicts: all basic info needed for any analysis
    """

    add_basic_analysis_info_gpt2(mt, ex)

    text_in = ex['text_in']
    struct_in = ex['struct_in']

    enc_sentence = ex['enc_sentence']
    tokenized_item = ex['tokenized_item']
    pre_sql_sequence = ex['pre_sql_sequence']

    text_range = ex['text_range']
    struct_range = ex['struct_range']
    sql_range = ex['sql_range']
    
    parsed_struct_in = ex['parsed_struct_in']

    alias2table = ex['alias2table']

    col2table = ex['col2table']
    col_name_counter = ex['col_name_counter']
    tab_name_counter = ex['tab_name_counter']

    token_ranges_dict = ex['struct_node_ranges_dict']

    sql_token_ranges = ex['sql_token_ranges']
    tok_ranges2type = ex['tok_ranges2type']
    type2tok_ranges = ex['type2tok_ranges']

    ## NOTE: Start specific code for creating node analysis samples
    # YS: Merging col and tab to remove "col & table" nodes when remove_dup = True,
    # Though now we can distinguish cols and tables, just keep the consistency of removing those 
    node_name_counter = col_name_counter + tab_name_counter

    if subject_type == 'column':
        node_name_ranges = token_ranges_dict['col_name_ranges']
        # node_name_counter = col_name_counter
    elif subject_type == 'table':
        node_name_ranges = token_ranges_dict['table_name_ranges']
        # node_name_counter = tab_name_counter
    elif subject_type == 'table_alias':
        _table_name_ranges = token_ranges_dict['table_name_ranges']
        node_name_ranges = {
            a : _table_name_ranges[alias2table[a]]  for a in alias2table.keys()
        }
        # node_name_counter = tab_name_counter
    else:
        raise NotImplementedError(subject_type)
    
    # sql_nodes = set()
    # for t in sql_tokens:
    #     if t in node_name_ranges:
    #         sql_nodes.add(t)
    
    sql_subject_ranges = []

    for s, e in list(type2tok_ranges[subject_type]):
        t = ex['seq_out'][s: e]

        if subject_type == 'table_alias':
            if t.endswith('.'):
                t = t[:-1]
                e -= 1      # YS NOTE: Added; why T5 didn't encounter errors here?
            node_name = alias2table[t]
        else:
            node_name = t
        
        _occ = node_name_counter[node_name]
        if _occ == 0:
            # Should have already raised SDRASampleError above!
            raise ValueError(struct_in, t, node_name)
        elif _occ > 1 and remove_struct_duplicate_nodes:
            continue

        sql_subject_ranges.append((s, e))
    
    # Add hardness info
    sql_hardness = evaluate_hardness(ex['seq_out'], ex['db_id'])

    analysis_ex_dicts =[]

    for s, e in sql_subject_ranges:
        node = expect = ex['seq_out'][s: e]
        # breakpoint()
        expect_input_ranges = node_name_ranges[node]    # this is input ranges, may be multiple
        dec_prompt = ex['seq_out'][:s].strip()
        node_role = detect_node_role(dec_prompt)
        # GPT2: add previous tokens to create full prompt
        dec_prompt = pre_sql_sequence + ' ' + dec_prompt

        ## Pulled out to `find_self_context_ranges()`
        # ## Processing for struct context; TODO: pull this part out (it's agnostic to `node`)
        # _all_node_range_lists = list(token_ranges_dict['col_name_ranges'].values()) + list(token_ranges_dict['table_name_ranges'].values()) + list(token_ranges_dict['db_id_ranges'].values())
        # _all_node_ranges = [rg
        #                     for rg_list in _all_node_range_lists
        #                     for rg in rg_list]
        # _all_left_endpoint = [s for s, e in _all_node_ranges] + [struct_range[1]]   # add start of non-struct
        # _all_right_endpoint = [e for s, e in _all_node_ranges] + [struct_range[0]]  # add end of non-struct
        # ## END TODO

        # context_range_endpoints = [struct_range[0]]
        # self_range_endpoints = []       # This is different from `expect_input_ranges`: this includes boundary toks
        # for tok_s, tok_e in expect_input_ranges:
        #     _l = max([e for e in _all_right_endpoint if e <= tok_s])
        #     _r = min([s for s in _all_left_endpoint if s >= tok_e])
        #     context_range_endpoints.extend([_l, _r])
        #     self_range_endpoints.extend([_l, _r])
        # context_range_endpoints.append(struct_range[1])

        # self_ranges = [(self_range_endpoints[i], self_range_endpoints[i+1])
        #                 for i in range(0, len(self_range_endpoints), 2)]

        # context_ranges = [(context_range_endpoints[i], context_range_endpoints[i+1])
        #                     for i in range(0, len(context_range_endpoints), 2)]
        # context_ranges = [(s, e) for s, e in context_ranges if e > s]       # rule out empty spans
        self_ranges, context_ranges = find_self_context_ranges(
            token_ranges_dict=token_ranges_dict,
            struct_range=struct_range,
            expect_input_ranges=expect_input_ranges)
        
        # We try to not corrupt the boundaries, keeping each node clean
        # When corrupting, use "expect_input_ranges" or "context_ranges" which do not include boundaries
        # When restoring, use "self_ranges" or "struct_no_node_tokens" (i.e. struct_range - node_range) which include boundaries
        
    
        # For table_aliases, the "as" prompts are not useful (e.g. ... join table_name as ___) since in Spider the
        # alias introductions are always monotonic, in the order of t1, t2, ... regardless of SQL
        if (subject_type == 'table_alias') and (dec_prompt.split()[-1] == 'as'):
            continue

        # Add role / text-match info
        if subject_type == 'column':
            node_table = col2table[node][0]
            text_match = check_col_text_match(ex, node, node_table)
        elif subject_type == 'table':
            text_match = check_table_text_match(ex, node)
        elif subject_type == 'table_alias':
            node_table = alias2table[node]
            text_match = check_table_text_match(ex, node_table)
        
        # Add node len directly in category
        node_len = len(mt.tokenizer.tokenize(expect))
        node_len_str = str(node_len) if node_len <= 3 else '4+'

        _ex = copy.deepcopy(ex)
        _ex['dec_prompt'] = dec_prompt
        _ex['expect'] = expect
        _ex['expect_type'] = subject_type
        _ex['remove_struct_duplicate_nodes'] = remove_struct_duplicate_nodes
        # _ex['parsed_struct_in'] = parsed_struct_in    # moved to front
        _ex['col2table'] = col2table
        _ex['token_ranges_dict'] = token_ranges_dict
        _ex['node_name_ranges'] = node_name_ranges
        _ex['expect_input_ranges'] = expect_input_ranges    # [(s, e), ...]
        # _ex['alias2table'] = alias2table  # moved to front
        _ex['self_ranges'] = self_ranges
        _ex['context_ranges'] = context_ranges

        _ex['category'] = {
            'sql_hardness': sql_hardness,
            'node_role': node_role,
            'text_match': text_match,
            'node_len': node_len_str,
        }

        analysis_ex_dicts.append(_ex)

    analysis_ex_dicts.sort(key=lambda d: d['dec_prompt'])
    return analysis_ex_dicts



def create_syntax_analysis_sample_dicts_gpt2(
        mt, 
        ex,
        include_literal=False,      # TODO: remove non-quoted literals (now only removing quoted ones)
        ):
    """
    BBB
    Create a new sample dict or analysis purpose, on syntax tokens 
    Return:
        analysis_ex_dicts: all basic info needed for any analysis
    """

    add_basic_analysis_info_gpt2(mt, ex)

    text_in = ex['text_in']
    struct_in = ex['struct_in']

    enc_sentence = ex['enc_sentence']
    tokenized_item = ex['tokenized_item']
    pre_sql_sequence = ex['pre_sql_sequence']

    text_range = ex['text_range']
    struct_range = ex['struct_range']
    
    parsed_struct_in = ex['parsed_struct_in']

    alias2table = ex['alias2table']

    col2table = ex['col2table']
    col_name_counter = ex['col_name_counter']
    tab_name_counter = ex['tab_name_counter']

    token_ranges_dict = ex['struct_node_ranges_dict']

    sql_token_ranges = ex['sql_token_ranges']
    tok_ranges2type = ex['tok_ranges2type']
    type2tok_ranges = ex['type2tok_ranges']

    ## NOTE: Start specific code for creating syntax analysis samples
    # struct_node_list = [f'{als}.' for als in alias2table.keys()] + list(alias2table.keys()) + list(node_name_counter.keys())

    # Add hardness info
    sql_hardness = evaluate_hardness(ex['seq_out'], ex['db_id'])
    
    analysis_ex_dicts = []

    for tok_char_span in type2tok_ranges['syntax']:
        s, e = tok_char_span
        expect = ex['seq_out'][s : e] 
        dec_prompt = ex['seq_out'][:s].strip()

        if (len(dec_prompt) == 0) or (expect == ';'):
            # first token is always "select"; also, empty prompt is prone to bugs
            continue

        # GPT2: add previous tokens to create full prompt
        dec_prompt = pre_sql_sequence + ' ' + dec_prompt

        _ex = copy.deepcopy(ex)
        _ex['dec_prompt'] = dec_prompt
        _ex['expect'] = expect
        _ex['expect_type'] = 'non_node'
        # _ex['in_quoted_literal'] = in_quoted_literal
        _ex['parsed_struct_in'] = parsed_struct_in
        _ex['col2table'] = col2table
        _ex['token_ranges_dict'] = token_ranges_dict
        _ex['alias2table'] = alias2table

        _ex['category'] = {
            'sql_hardness': sql_hardness,
        }

        analysis_ex_dicts.append(_ex)

        # result['expect'] = node      # actually already available in ['answer']
        # result['subject_type'] = subject_type
        # if subject_type == 'column':
        #     result['table'] = col2table[node][0]    # col2table[node] is a list
        # elif subject_type == 'table':
        #     result['table'] = node
        # result['db_id'] = ex['db_id']
        # result['expect_input_indices'] = enc_token_range
    return analysis_ex_dicts



def add_clean_prediction_gpt2(
        mt,
        a_ex,
        samples=10,
        device='cuda'):
    """
    Called after `create_analysis_sample_dicts()` to add keys about clean prediction 
    Args:
        a_ex (Dict): analysis_ex, output from `create_analysis_sample_dicts()`
    """

    a_ex = copy.deepcopy(a_ex)

    # enc_sentence = a_ex['enc_sentence']
    dec_prompt = a_ex['dec_prompt']
    expect = a_ex['expect']

    inp = make_inputs_gpt2(
        mt.tokenizer,
        [dec_prompt] * (samples + 1),
        answer=expect,
        device=device
    )
    answer_len = 1
    if expect is not None:
        answer_len = len(mt.tokenizer.tokenize(expect))
    with torch.no_grad():
        answers_t, base_score = [d[0] for d in predict_from_input_uskg_gpt2_multi_token(mt.model, inp, pred_len=answer_len)]
    base_score = base_score.min().item()
    # [answer] = decode_tokens(mt.tokenizer, [answer_t])
    answer = decode_sentences(mt.tokenizer, answers_t)
    is_correct_pred = (answer.strip() == expect)

    # a_ex['inp'] = inp     # might take GPU mem...
    a_ex['answer_len'] = answer_len
    a_ex['base_score'] = base_score
    a_ex['answers_t'] = answers_t
    a_ex['answer'] = answer
    a_ex['correct_prediction'] = is_correct_pred

    return a_ex

## Can just use ctu.make_basic_result_dict
# def make_basic_result_dict_gpt2(a_ex):
#     pass

def create_analysis_sample_dicts_all_nodes_gpt2(
        mt,
        ex,
        remove_struct_duplicate_nodes=True):
    
    """
    BBB
    Create a new a_ex for analysis purpose on all nodes; save the node used/unused
    Return:
        a_ex (Dict): all basic info needed for any analysis
    """

    # a_ex_col_list = create_analysis_sample_dicts(
    #                 mt, 
    #                 ex,
    #                 subject_type='column',
    #                 remove_struct_duplicate_nodes=remove_struct_duplicate_nodes)
    # a_ex_tab_list = create_analysis_sample_dicts(
    #                 mt,
    #                 ex,
    #                 subject_type='table',
    #                 remove_struct_duplicate_nodes=remove_struct_duplicate_nodes)     
    
    # Consistent with current setting, remove "col+tab" duplicates if set to True

    # a_ex_list = a_ex_col_list + a_ex_tab_list


    add_basic_analysis_info_gpt2(mt, ex)

    text_in = ex['text_in']
    struct_in = ex['struct_in']

    enc_sentence = ex['enc_sentence']
    # enc_tokenized = ex['enc_tokenized']

    text_range = ex['text_range']
    struct_range = ex['struct_range']
    
    parsed_struct_in = ex['parsed_struct_in']

    alias2table = ex['alias2table']

    col2table = ex['col2table']
    col_name_counter = ex['col_name_counter']
    tab_name_counter = ex['tab_name_counter']

    token_ranges_dict = ex['struct_node_ranges_dict']

    sql_token_ranges = ex['sql_token_ranges']
    tok_ranges2type = ex['tok_ranges2type']
    type2tok_ranges = ex['type2tok_ranges']

    ## NOTE: Start specific code for creating analysis samples for all nodes (no specific nodes)
    node_name_counter = col_name_counter + tab_name_counter

    sql_col_nodes = ex['sql_col_nodes']
    sql_tab_nodes = ex['sql_tab_nodes']

    occ_cols = []
    occ_tabs = []

    for node_name in sql_col_nodes:
        _occ = node_name_counter[node_name]
        if _occ == 0:
            # Should have already raised SDRASampleError above!
            raise ValueError(struct_in, node_name, 'column')
        elif _occ > 1 and remove_struct_duplicate_nodes:
            continue
        occ_cols.append(node_name)

    for node_name in sql_tab_nodes:
        _occ = node_name_counter[node_name]
        if _occ == 0:
            # Should have already raised SDRASampleError above!
            raise ValueError(struct_in, node_name, 'table')
        elif _occ > 1 and remove_struct_duplicate_nodes:
            continue
        occ_tabs.append(node_name)

    if len(occ_cols) + len(occ_tabs) == 0:
    # if len(a_ex_list) == 0:
        raise SDRASampleError('No available nodes')

    a_ex = copy.deepcopy(ex)
    # del a_ex['expect']
    # del a_ex['expect_input_ranges']
    # del a_ex['self_ranges']
    # del a_ex['context_ranges']
    # del a_ex['expect_type']
    # del a_ex['node_name_ranges']
    # del a_ex['category']['node_role']
    # del a_ex['category']['text_match']
    # del a_ex['category']['node_len']
    a_ex['dec_prompt'] = 'select'       # placeholder

    # already added
    # token_ranges_dict = a_ex['token_ranges_dict']
    # text_range = a_ex['text_range']
    # struct_range = a_ex['struct_range']

    # # BUG: can have duplicates here!
    # occ_cols = [d['expect'] for d in a_ex_col_list]
    # occ_tabs = [d['expect'] for d in a_ex_tab_list]
    
    col_name_ranges = a_ex['struct_node_ranges_dict']['col_name_ranges']
    table_name_ranges = a_ex['struct_node_ranges_dict']['table_name_ranges']

    non_occ_cols = [col for col in col_name_ranges
                        if (col not in occ_cols) and not (remove_struct_duplicate_nodes and len(col_name_ranges[col]) > 1)]
    non_occ_tabs = [tab for tab in table_name_ranges
                        if (tab not in occ_tabs) and not (remove_struct_duplicate_nodes and len(table_name_ranges[tab]) > 1)]

    a_ex['occ_cols'] = occ_cols
    a_ex['occ_tabs'] = occ_tabs
    a_ex['non_occ_cols'] = non_occ_cols
    a_ex['non_occ_tabs'] = non_occ_tabs

    a_ex['col_self_ranges'] = dict()
    a_ex['col_context_ranges'] = dict()
    a_ex['tab_self_ranges'] = dict()
    a_ex['tab_context_ranges'] = dict()

    ## Processing for struct context
    _all_node_range_lists = list(token_ranges_dict['col_name_ranges'].values()) + list(token_ranges_dict['table_name_ranges'].values()) + list(token_ranges_dict['db_id_ranges'].values())
    _all_node_ranges = [rg
                        for rg_list in _all_node_range_lists
                        for rg in rg_list]
    _all_left_endpoint = [s for s, e in _all_node_ranges] + [struct_range[1]]   # add start of non-struct
    _all_right_endpoint = [e for s, e in _all_node_ranges] + [struct_range[0]]  # add end of non-struct

    for col in occ_cols + non_occ_cols:
        expect_input_ranges = token_ranges_dict['col_name_ranges'][col]

        context_range_endpoints = [struct_range[0]]
        self_range_endpoints = []       # This is different from `expect_input_ranges`: this includes boundary toks
        for tok_s, tok_e in expect_input_ranges:
            _l = max([e for e in _all_right_endpoint if e <= tok_s])
            _r = min([s for s in _all_left_endpoint if s >= tok_e])
            context_range_endpoints.extend([_l, _r])
            self_range_endpoints.extend([_l, _r])
        context_range_endpoints.append(struct_range[1])

        self_ranges = [(self_range_endpoints[i], self_range_endpoints[i+1])
                        for i in range(0, len(self_range_endpoints), 2)]

        context_ranges = [(context_range_endpoints[i], context_range_endpoints[i+1])
                            for i in range(0, len(context_range_endpoints), 2)]
        context_ranges = [(s, e) for s, e in context_ranges if e > s]       # rule out empty spans

        a_ex['col_self_ranges'][col] = self_ranges
        a_ex['col_context_ranges'][col] = context_ranges


    for tab in occ_tabs + non_occ_tabs:
        expect_input_ranges = token_ranges_dict['table_name_ranges'][tab]

        context_range_endpoints = [struct_range[0]]
        self_range_endpoints = []       # This is different from `expect_input_ranges`: this includes boundary toks
        for tok_s, tok_e in expect_input_ranges:
            _l = max([e for e in _all_right_endpoint if e <= tok_s])
            _r = min([s for s in _all_left_endpoint if s >= tok_e])
            context_range_endpoints.extend([_l, _r])
            self_range_endpoints.extend([_l, _r])
        context_range_endpoints.append(struct_range[1])

        self_ranges = [(self_range_endpoints[i], self_range_endpoints[i+1])
                        for i in range(0, len(self_range_endpoints), 2)]

        context_ranges = [(context_range_endpoints[i], context_range_endpoints[i+1])
                            for i in range(0, len(context_range_endpoints), 2)]
        context_ranges = [(s, e) for s, e in context_ranges if e > s]       # rule out empty spans

        a_ex['tab_self_ranges'][tab] = self_ranges
        a_ex['tab_context_ranges'][tab] = context_ranges

    return a_ex


def main_sdra_1_struct_node_restore(args):
    """
    Exp 1: Corrupt the struct node of interest, restore every single state
    Purpose: check where is the node-relevant info stored
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'dev_{args.subject_type}_{args.part}-tmp'
    result_save_dir = os.path.join(args.result_dir, 'exp1_struct_node_restore')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    start_id = 0
    end_id = n_ex
    stride = 50
    with open(result_save_path, 'w') as f:
        for i in tqdm(range(start_id, end_id, stride), desc=f"Main loop: {exp_name}", ascii=True):
            ex = processed_spider_dataset[i]
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


def main_sdra_2_text_struct_interaction(args):
    """ 
    Exp2 (text struct interaction): corrupt text (or struct/all), restore the *final* encoding
        of different parts to check recovery effect
    Adapted from notebook.
    """

    out_dir = os.path.join(args.result_dir, 'exp2_text_struct_interaction')
    os.makedirs(out_dir, exist_ok=True)
    exp_name = f'exp=2_{args.ds}_{args.subject_type}'
    res_save_path = os.path.join(out_dir, f'{exp_name}.jsonl')

    # Load model and dataset 
    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    # Stats
    total_samples = 0
    n_good_samples = 0
    n_too_hard = 0      # incorrect
    n_too_easy = 0      # base > 0.5, base - low < 0.5

    base_scores = []
    low_scores = []

    ## len = n_good_samples
    restore_scores_dict = {
        'text': [],
        'struct': [],
        'node': [],
        'struct_no_node': [],
        'node_corrupt_all': [],
        ## 2.0.1: cancelled
        # 'ctname': [],      # col name + table name (col belongs to) 
        # 'catname': [],     # col name + all table names 
        # 'full_table': [],  # full table (col belongs to) 
        # 'all_col': [],     # all col names (regardless of table)
    }

    ## len = total_samples
    mutual_scores_dict = {
        f'{text}-{struct}': []
        for text in ['clean_t', 'dc_t', 'dirty_t']
        for struct in ['clean_s', 'dc_s', 'dirty_s']
    }

    f = open(res_save_path, 'w')
    start_id = 0
    end_id = n_ex
    stride = 1
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        # DEBUG
        # print(f'** DEBUG info:')
        # print(f'** ex_id = {ex_id}')
        # _mem_r = torch.cuda.memory_reserved(0)
        # _mem_a = torch.cuda.memory_allocated(0)
        # print(f'** mem reserved: {_mem_r}')
        # print(f'** mem allocated: {_mem_a}')

        ex = processed_spider_dataset[ex_id]
        
        analysis_samples = create_analysis_sample_dicts(
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
                ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp2_text_struct(mt_uskg, a_ex)

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            else:
                n_too_hard += (not result['correct_prediction'])
                n_too_easy += (result.get('base_score', 0.0) - result.get('low_score', 0.0) < 0.5)

            if result['correct_prediction']:
                for k in mutual_scores_dict:
                    mutual_scores_dict[k].append(result['mutual_scores'][k])

            if result['is_good_sample']:
                base_scores.append(result['base_score'])
                low_scores.append(result['low_score'])
                restore_scores_dict['text'].append(result['r_text_score'])
                restore_scores_dict['struct'].append(result['r_struct_score'])
                restore_scores_dict['node'].append(result['r_node_score'])
                restore_scores_dict['struct_no_node'].append(result['r_struct_no_node_score'])
                restore_scores_dict['node_corrupt_all'].append(result['r_node_corrupt_all_score'])
            
        ex_out_dict = {
            'ex_id': ex_id,
            'trace_results': ex_results,
        }
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print(total_samples, n_good_samples, n_too_hard, n_too_easy)
    print()

    print("-"*10 + "Results for exp2" + "-"*10)
    print(f'Score\tavg_gain\tperc_recover')
    for score_label, high_scores in [
        ('base_scores', base_scores),
        ('restore_text_scores', restore_scores_dict['text']),
        ('restore_struct_scores', restore_scores_dict['struct']),
        ('restore_node_scores', restore_scores_dict['node']),
        ('restore_struct_no_node_scores', restore_scores_dict['struct_no_node']),
        ('restore_node_corrupt_all_scores', restore_scores_dict['node_corrupt_all']),
    ]:
        avg_gain = numpy.mean([h - l for h, l in zip(high_scores, low_scores)])
        perc_recover = numpy.mean([h - l > 0.5 for h, l in zip(high_scores, low_scores)])
        print(f'{score_label}\t{avg_gain:.4f}\t{perc_recover:.4f}')
    print()

    print("-"*10 + "Results for exp2.1.1" + "-"*10)
    msg = ' '*8
    for k_t in ['clean_t', 'dc_t', 'dirty_t']:
        msg += f'{k_t:8s}'
    msg += '\n'
    for k_s in ['clean_s', 'dc_s', 'dirty_s']:
        msg += f'{k_s:8s}'
        for k_t in ['clean_t', 'dc_t', 'dirty_t']:
            k = f'{k_t}-{k_s}'
            scores = mutual_scores_dict[k]
            avg = numpy.mean(scores)
            msg += f'{avg:.4f}  '
        msg += '\n'
    print(msg)


def main_sdra_2_2_dirty_text_struct_restore(args):
    """
    AAA
    Exp 2.2: Corrupt the text, restoring the struct node of interest
    Purpose: check the pos of contextualization of text into struct nodes
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=2.2_{args.ds}_{args.subject_type}'
    result_save_dir = os.path.join(args.result_dir, 'exp2.2_dirty_text_struct_restore')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    start_id = 0
    end_id = n_ex
    stride = 1
    with open(result_save_path, 'w') as f:
        for i in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
            ex = processed_spider_dataset[i]
            results = trace_section_corrupt_restore(
                mt=mt_uskg,
                ex=ex,
                subject_type=args.subject_type,
                replace=True,
                # part=args.part,
                part='encoder'
            )
            # TODO: trace_results -> col_trace_results, tab_trace_results
            dump_dict = dict(
                ex_id=i,
                trace_results=results,
            )
            f.write(json.dumps(dump_dict, indent=None) + '\n')


def main_sdra_2_3_section_corruption_effect(args):
    """
    Exp 2.3: section corruption effect
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    # args.exp_id = "2.3"
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

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
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
            # analysis_samples = create_syntax_analysis_sample_dicts(mt_uskg, ex)
        else:
            analysis_samples = create_analysis_sample_dicts(
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

            a_ex = add_clean_prediction(mt_uskg, a_ex)
            
            # exp_func = trace_exp2_3
            result = trace_exp2_3(
                mt_uskg,
                a_ex,
                replace=args.replace,
                noise=args.noise,
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



def main_sdra_3_0_node_corrupt_effect(args):
    """
    Exp 3.0: Corrupt each node, see if it has effect on prediction
    Purpose: (expanding exp0) see whether / when other struct nodes are needed
    """
    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    result_save_dir = os.path.join(args.result_dir, 'exp3_relational_nodes_mutual')
    os.makedirs(result_save_dir, exist_ok=True)
    exp_name = f'exp=3.0_{args.ds}_{args.subject_type}'
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    # Task-specific args
    remove_struct_duplicate_nodes = True
    skips = ('token',)

    start_id = 0
    end_id = n_ex
    stride = 1
    with open(result_save_path, 'w') as f:
        for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
            ex = processed_spider_dataset[ex_id]

            analysis_samples = create_analysis_sample_dicts(
                mt_uskg, ex, args.subject_type,
                remove_struct_duplicate_nodes=remove_struct_duplicate_nodes)

            for a_ex in analysis_samples:
                
                # TODO: check input_len, skip if > 500
                a_ex = add_clean_prediction(mt_uskg, a_ex)

                result_d = token_corruption_influence_uskg(
                    mt_uskg,
                    # enc_sentence=enc_sentence,
                    # dec_prompt=dec_prompt,
                    # expect=expect,
                    a_ex,
                    replace=True,
                    use_tqdm=False,
                    skips=skips,
                )
                
                result_d['ex_id'] = ex_id
                # if args.subject_type == 'column':
                #     result_d['expect_table'] = col2table[expect][0]
                # elif args.subject_type == 'table':
                #     result_d['expect_table'] = expect

                # all_results.append(result_d)
                f.write(json.dumps(result_d, indent=None) + '\n')


def main_sdra_3_1_dirty_struct_context_restore(args):
    """
    AAA
    Exp 3.1: Corrupt the full struct context (everything but the node of interest & surronding structure indicators like "|", ":", etc.) ; restore the representation of context (per layer / layer range) and check repair effect
    Purpose: check the process of contextualization of all context tokens into node of interest
    """

    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=3.1_{args.ds}_{args.subject_type}-tmp'
    result_save_dir = os.path.join(args.result_dir, 'exp3.1_dirty_struct_context_restore')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)


    # TODO: Stats
    total_samples = 0
    n_good_samples = 0
    n_too_hard = 0      # incorrect
    n_too_easy = 0      # base > 0.5, base - low < 0.5

    base_scores = []
    low_scores = []

    ## len = n_good_samples
    restore_scores_dict = {
        # 'text': [],
        # 'struct': [],
        # 'node': [],
        # 'struct_no_node': [],
        # 'node_corrupt_all': [],
    }

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    # end_id = 50
    stride = 111
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = create_analysis_sample_dicts(
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
                ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp3_1_struct_context_restore(mt_uskg, a_ex)

            ex_results.append(result)

            total_samples += 1
            if result['is_good_sample']:
                n_good_samples += 1
            else:
                n_too_hard += (not result['correct_prediction'])
                n_too_easy += (result.get('base_score', 0.0) - result.get('low_score', 0.0) < 0.5)

            # if result['correct_prediction']:
            #     for k in mutual_scores_dict:
            #         mutual_scores_dict[k].append(result['mutual_scores'][k])

            if result['is_good_sample']:
                # TODO: global results registration
                pass
                # base_scores.append(result['base_score'])
                # low_scores.append(result['low_score'])
                # restore_scores_dict['text'].append(result['r_text_score'])
                # restore_scores_dict['struct'].append(result['r_struct_score'])
                # restore_scores_dict['node'].append(result['r_node_score'])
                # restore_scores_dict['struct_no_node'].append(result['r_struct_no_node_score'])
                # restore_scores_dict['node_corrupt_all'].append(result['r_node_corrupt_all_score'])
            
        ex_out_dict = {
            'ex_id': ex_id,
            'trace_results': ex_results,
        }
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print(total_samples, n_good_samples, n_too_hard, n_too_easy)
    print()

    print("-"*10 + "Results for exp3.1" + "-"*10)
    print(f'Score\tavg_gain\tperc_recover')
    print(f'TODO')


def main_sdra_3_2_dirty_struct_context_compare(args):
    """
    AAA
    Exp 3.2: Corrupt the full struct context vs. no corruption
    Purpose: checking the effect of struct context corruption, is it more positive (denoising) or negative (removing info)?
    """

    spider_dataset_path = args.spider_dataset_path
    spider_db_dir = args.spider_db_dir
    data_cache_dir = args.data_cache_dir

    exp_name = f'exp=3.2_{args.ds}_{args.subject_type}'
    result_save_dir = os.path.join(args.result_dir, 'exp3.2_dirty_struct_context_compare')
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f'{exp_name}.jsonl')

    mt_uskg = ModelAndTokenizer_USKG('t5-large-prefix')

    processed_spider_dataset = load_spider_dataset(args, mt_uskg)
    n_ex = len(processed_spider_dataset)

    # Stats
    total_samples = 0
    
    ## len = n_good_samples
    compare_results_counter = Counter()

    f = open(result_save_path, 'w')
    start_id = 0
    end_id = n_ex
    stride = 1
    # with open(result_save_path, 'w') as f:
    for ex_id in tqdm(range(start_id, end_id, stride), desc=f"MAIN: {exp_name}", ascii=True):
        ex = processed_spider_dataset[ex_id]

        analysis_samples = create_analysis_sample_dicts(
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
                ex_out_dict['trace_results'] = []
                ex_out_dict['err_msg'] = f'Input too long: {input_len} > 500'
                input_too_long = True

        ex_results = []
        for a_ex in analysis_samples:
            if input_too_long:
                continue

            a_ex = add_clean_prediction(mt_uskg, a_ex)
            
            result = trace_exp3_2_struct_context_corrupt_compare(mt_uskg, a_ex)

            ex_results.append(result)

            total_samples += 1

            # global results registration
            compare_results_counter[result['compare']['correctness_compare']] += 1
            
        ex_out_dict = {
            'ex_id': ex_id,
            'trace_results': ex_results,
        }
        f.write(json.dumps(ex_out_dict, indent=None) + '\n')
    f.close()

    print(total_samples)
    print()

    print("-"*10 + "Results for exp3.2" + "-"*10)
    print(compare_results_counter)



EXP_CONFIGS = {
    '2.3.0': {
        'replace': True,
        'noise': 0.1,
    },
    '2.3.1': {
        'replace': True,
        'noise': 0.0,
    },
    # '2.3.2': {
    #     'replace': False,
    #     'noise': 0.1,
    # },
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

    # args = Namespace()
    args.ds = 'dev'                     # train, dev
    # args.subject_type = 'column'         # table, table_alias, column, (value)
    args.part = 'encoder'               # encoder, decoder, both
    args.spider_dataset_path = f'/home/yshao/Projects/SDR-analysis/data/spider/{args.ds}+ratsql_graph.json'
    args.spider_db_dir = '/home/yshao/Projects/language/language/xsp/data/spider/database'
    args.data_cache_dir = '/home/yshao/Projects/rome/cache'
    args.spider_tables_path = '/home/yshao/Projects/language/language/xsp/data/spider/tables.json'

    args.result_dir = '/home/yshao/Projects/rome/results'

    evaluate_hardness.evaluator = load_evaluator(args)

    # args.is_tmp = False

    args.result_save_dir = os.path.join(args.result_dir, "exp2.3_section_corruption_effect")

    args.subject_type = 'column'
    main_sdra_2_3_section_corruption_effect(args)

    args.subject_type = 'table'
    main_sdra_2_3_section_corruption_effect(args)

    args.subject_type = 'table_alias'
    main_sdra_2_3_section_corruption_effect(args)


if __name__ == "__main__":
    main()
