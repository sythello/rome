import argparse
import json
import os
import sys
import re
from collections import defaultdict
import copy
import string

import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from dsets import KnownsDataset
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally

from util import uskg

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

USKG_GPT2_OUT_SPLITTER = '; SQL:'
USKG_GPT2_OUT_SPLITTER_CHARS = ';SQL:'

USKG_GPT2_FINALIZER = 'END OF SQL'      # keep the ";" to assure correct ending
USKG_GPT2_FINALIZER_CHARS = 'ENDOFSQL'

def run_model_forward_uskg_gpt2(
    model,
    input_ids,
    attention_mask,
    # decoder_input_ids,
    # decoder_attention_mask,
    output_attentions=False,
    labels=None,
):
    bsz = input_ids.shape[0]

    kwargs = dict()
    description_representation = model.get_description_representation(kwargs)
    knowledge_representation = model.get_knowledge_representation(kwargs)

    past_prompt = model.get_prompt(
        bsz=bsz, description=description_representation, knowledge=knowledge_representation,
    )

    model_out = model.pretrain_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # decoder_input_ids=decoder_input_ids,
        # decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_attentions=output_attentions,
        past_prompt=past_prompt
    )

    return model_out


class ModelAndTokenizer_USKG_GPT2:
    """
    An object to hold on to (or automatically download and hold)
    a USKG model and tokenizer.  Counts the number of layers.
    """

    def __init__(
        self,
        model_name=None,
        # model=None,
        # tokenizer=None,
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

        # if (model is None) or (tokenizer is None):
        assert model_name is not None
        model, tokenizer_uskg, tokenizer_fast, training_args, model_args, task_args = uskg.load_model_uskg(model_name, untie_embeddings=False)
        model = model.eval().to(device=device)

        self.tokenizer = tokenizer_fast
        self.tokenizer_uskg = tokenizer_uskg
        self.model = model
        self.training_args = training_args
        self.model_args = model_args
        self.task_args = task_args
        # Layer names in model:
        # 'pretrain_model.transformer',
        # 'pretrain_model.transformer.wte',
        # 'pretrain_model.transformer.wpe',
        # 'pretrain_model.transformer.drop',
        # 'pretrain_model.transformer.h',
        # 'pretrain_model.transformer.h.0',
        # 'pretrain_model.transformer.h.0.ln_1',
        # 'pretrain_model.transformer.h.0.attn',
        # 'pretrain_model.transformer.h.0.attn.c_attn',
        # 'pretrain_model.transformer.h.0.attn.c_proj',
        # 'pretrain_model.transformer.h.0.attn.attn_dropout',
        # 'pretrain_model.transformer.h.0.attn.resid_dropout',
        # 'pretrain_model.transformer.h.0.ln_2',
        # 'pretrain_model.transformer.h.0.mlp',
        # 'pretrain_model.transformer.h.0.mlp.c_fc',
        # 'pretrain_model.transformer.h.0.mlp.c_proj',
        # 'pretrain_model.transformer.h.0.mlp.dropout',

        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^pretrain_model\.transformer\.h\.\d+$", n))
        ]

        # self.num_enc_layers = len(self.enc_layer_names)
        # self.num_dec_layers = len(self.dec_layer_names)
        self.num_layers = len(self.layer_names)
        

    def __repr__(self):
        return (
            f"ModelAndTokenizer_USKG(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername_uskg_gpt2(model, num, kind=None):
    """
    num: layer number
    kind: embed / attn / mlp / None
    """

    # assert hasattr(model.pretrain_model, part), f"{part} not in the model.pretrain_model of type {type(model.pretrain_model)}"

    # Original ROME code modified (in original code, model is plain gpt2; here, model.pretrained_model is gpt2)
    if hasattr(model.pretrain_model, "transformer"):
        if kind == "embed":
            return "pretrain_model.transformer.wte"
        return f'pretrain_model.transformer.h.{num}{"" if kind is None else "." + kind}'
    # if hasattr(model, "gpt_neox"):
    #     if kind == "embed":
    #         return "gpt_neox.embed_in"
    #     if kind == "attn":
    #         kind = "attention"
    #     return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"

    # Removed USKG-T5 code
    # # Layer names
    # # decoder.block.2
    # # decoder.block.2.layer.0.SelfAttention
    # # decoder.block.2.layer.1.EncDecAttention
    # # decoder.block.2.layer.2.DenseReluDense

    # if kind == "embed":
    #     return f"pretrain_model.{part}.embed_tokens"

    # _kind = None
    # if kind == "self_attn":
    #     _kind = "layer.0.SelfAttention"
    # elif kind == "cross_attn":
    #     assert part == "decoder", f"model {part} doesn't have {kind}"
    #     _kind = "layer.1.EncDecAttention"
    # elif kind == "mlp":
    #     if part == "encoder":
    #         _kind = "layer.1.DenseReluDense"
    #     elif part == "decoder":
    #         _kind = "layer.2.DenseReluDense"
    
    # return f"pretrain_model.{part}.block.{num}{'' if _kind is None else '.' + _kind}"


# def make_dec_prompt(dec_target, subject):
#     # YS TODO: need to update?
#     dec_target = ' ' + dec_target + ' '  # to avoid the matching problems at the ends 
#     # m = re.search(fr'\W({subject})\W', dec_target)
#     m_iter = re.finditer(fr'\W({subject})\W', dec_target)
#     prompts = []
#     for m in m_iter:
#         s, e = m.span(1)
#         prompt = dec_target[:s].strip()
#         prompts.append(prompt)

#     assert prompts, (dec_target, subject)
    
#     return prompts


def find_text_struct_in_range_gpt2(tokenizer, token_array):
    toks = uskg.decode_tokens(tokenizer, token_array)
    return find_text_struct_in_range_str_tokens_gpt2(toks)


def find_text_struct_in_range_str_tokens_gpt2(toks):
    # YS NOTE: using this char-level implementation to avoid weird tokenizations, with overlapping tokens in text/splitter/struct
    # Also, assuming no padding toks
    # Spans are [l, r)
    # YS NOTE: GPT2 decoding single tokens still maintains their spaces in front, so when concat back it's the same as original!
    whole_string = "".join(toks)
    ts_splitter_char_loc = whole_string.index(uskg.USKG_SPLITTER)
    io_splitter_char_loc = whole_string.index(USKG_GPT2_OUT_SPLITTER)
    finalizer_char_loc = whole_string.index(USKG_GPT2_FINALIZER)
    loc = 0
    text_tok_end, struct_tok_start, struct_tok_end, out_tok_start, out_tok_end = None, None, None, None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if text_tok_end is None and loc > ts_splitter_char_loc:
            text_tok_end = i
        elif struct_tok_start is None and loc > ts_splitter_char_loc + len(uskg.USKG_SPLITTER):
            struct_tok_start = i
        elif struct_tok_end is None and loc >= io_splitter_char_loc:
            struct_tok_end = i
        elif out_tok_start is None and loc > io_splitter_char_loc + len(USKG_GPT2_OUT_SPLITTER):
            out_tok_start = i
        elif out_tok_end is None and loc >= finalizer_char_loc:
            out_tok_end = i
            break

    # return (0, text_tok_end), (struct_tok_start, len(toks) - 1)     # the last token is </s>, don't corrupt it
    return (0, text_tok_end), (struct_tok_start, struct_tok_end), (out_tok_start, out_tok_end)

def find_struct_name_ranges_gpt2(tokenizer, ex):
    """ 
    Need ex keys:
        struct_in, tokenized_item
    (Need to pass struct_in since it's easier to parse than the string reconstructed by tokenizer)
    """

    # text_in = ex['text_in']
    struct_in = ex['struct_in']
    # if 'enc_sentence' not in ex:
    #     ex['enc_sentence'] = f"{text_in}; structed knowledge: {struct_in}"
    
    # sentence = decode_sentences(tokenizer, token_array)
    # assert USKG_SPLITTER in sentence, f'"{sentence}" does not have splitter {USKG_SPLITTER}'
    # # select the struct_in part
    if ('text_range' not in ex) or ('struct_range' not in ex) or ('sql_range' not in ex):
        token_array = ex['tokenized_item']['input_ids']
        text_range, struct_range, sql_range = find_text_struct_in_range_gpt2(tokenizer, token_array)
        ex['text_range'] = text_range
        ex['struct_range'] = struct_range
        ex['sql_range'] = sql_range
    else:
        text_range = ex['text_range']
        struct_range = ex['struct_range']
        sql_range = ex['sql_range']
    
    sb, se = struct_range
    # token_array = token_array[sb : se]
    # struct_in = ' '.join(decode_tokens(tokenizer, token_array))

    if 'parsed_struct_in' not in ex:
        parsed_struct_in = uskg.parse_struct_in(struct_in)
        ex['parsed_struct_in'] = parsed_struct_in
    else:
        parsed_struct_in = ex['parsed_struct_in']

    db_id_t, tables = parsed_struct_in
    stk = tokenizer(struct_in)
    s_full_word_char_ranges = []    # [i]: struct_in.split()[i] -> (char_st, char_ed)
    _c = 0
    for i, w in enumerate(struct_in.split(' ')):
        # Here splitting by ' ' instead of default splitting, to work around multi-space problems
        try:
            assert struct_in[_c : _c + len(w)] == w
        except AssertionError:
            breakpoint()
        if len(w) > 0:
            s_full_word_char_ranges.append((_c, _c + len(w)))
        _c += len(w) + 1

    db_id_ranges = defaultdict(list)
    table_name_ranges = defaultdict(list)      # dict[name, [(st, ed)]]; same for others
    col_name_ranges = defaultdict(list)
    val_name_ranges = defaultdict(list)

    # YS NOTE: the definition of "words" are different in each tokenizer
    # In T5, it's splitted by spaces; in GPT2, it's also splitted by puncts
    # For uniformity, we'd better use chars, w_idx -> char_idx -> tokens
    def _add_t(name_t, d, extra_name_t=None):
        # """
        # extra_name_t: for column, provide the value tuples to add their range to column
        # """
        nonlocal stk, s_full_word_char_ranges
        nonlocal sb, se
        w_idx, name, full_str = name_t
        # full_words = full_str.split()
        # w_idx_last = w_idx + len(full_words) - 1
        # s, _ = stk.word_to_tokens(w_idx)
        # _, e = stk.word_to_tokens(w_idx_last)
        char_s = s_full_word_char_ranges[w_idx][0]
        char_e = char_s + len(full_str) - 1
        s = stk.char_to_token(char_s)
        e = stk.char_to_token(char_e) + 1
        # print('* _add_t()')
        # print('** name_t:', name_t)
        # print('** w_idx:', w_idx)
        # print('** s, e:', s, e)
        d[name].append((s + sb, e + sb))

    _add_t(db_id_t, db_id_ranges)
    for table_name_t, cols in tables:
        _add_t(table_name_t, table_name_ranges)
        for col_name_t, vals in cols:
            _add_t(col_name_t, col_name_ranges)
            for val_t in vals:
                _add_t(val_t, val_name_ranges)

    struct_node_ranges_dict = dict(
        db_id_ranges=db_id_ranges,
        table_name_ranges=table_name_ranges,
        col_name_ranges=col_name_ranges,
        val_name_ranges=val_name_ranges,
    )

    ex['struct_node_ranges_dict'] = struct_node_ranges_dict

    return struct_node_ranges_dict

