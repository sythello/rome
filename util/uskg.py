import argparse
import json
import os
import sys
import re
from collections import defaultdict
import copy
import string

import numpy
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

from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)

# from uskg.models.unified.prefixtuning import Model
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


USKG_SPLITTER = '; structed knowledge: '
USKG_SPLITTER_CHARS = ';structedknowledge:'

def load_model_uskg(model_name, untie_embeddings=False):
    save_argv = sys.argv

    if model_name == 't5-large-prefix':
        uskg_config = '/home/yshao/Projects/UnifiedSKG/configure/Salesforce/A-T5_large_prefix_spider_with_cell_value.cfg'
        model_path = 'hkunlp/from_all_T5_large_prefix_spider_with_cell_value2'
    elif model_name == 't5-base-prefix':
        uskg_config = '/home/yshao/Projects/UnifiedSKG/configure/Salesforce/A-T5_base_prefix_spider_with_cell_value.cfg'
        model_path = 'hkunlp/from_all_T5_base_prefix_spider_with_cell_value2'
    else:
        raise NotImplementedError(model_name)

    # Set args here for runnning on notebook, we make them out here to make it more illustrative.
    sys.argv = ['/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py',  # This is the name of your .py launcher when you run this line of code.
                # belows are the parameters we set, take spider for example
                '--cfg', uskg_config,
                '--output_dir', './tmp']
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # model_args = Configure.Get(training_args.cfg)
    model_args = Configure.get_file_cfg(training_args.cfg)      # Get will append 'configure' dir
    task_args_path = dict(list(model_args.arg_paths))['spider']
    task_args = Configure.get_file_cfg(task_args_path)

    sys.argv = save_argv

    # Tokenizer: 'fast' for word/token/char mapping functions
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print('Using tokenizer:', model_args.bert.location)
    tokenizer_fast = AutoTokenizer.from_pretrained(
        model_args.bert.location, use_fast=True)

    # Model: for model_path, now support: USKG (hkunlp/xxx); original T5 (t5-xxx); random T5 (t5-xxx-rd)
    # model_path = main_args.model_path
    if model_path.startswith('hkunlp'):
        # USKG
        if 'prefix' in model_path:
            assert 'prefix' in uskg_config, ('Mismatch',
                                             model_path, uskg_config)
            model = prefixtuning.Model(model_args)
        elif 'finetune' in model_path:
            assert 'finetune' in uskg_config, ('Mismatch',
                                               model_path, uskg_config)
            model = finetune.Model(model_args)
        else:
            raise ValueError(model_path)

        model.load(model_path)

    elif model_path.startswith('t5'):
        model = finetune.Model(model_args)
        assert model_path.startswith(model_args.bert.location), (
            'Mismatch', model_path, model_args.bert.location)  # check USKG & T5 version consistency
        if model_path.endswith('rd'):
            # random T5
            model.pretrain_model.init_weights()
        else:
            # original T5, already loaded
            pass
    else:
        raise ValueError(model_path)
    
    if untie_embeddings:
        # TODO: more memory-efficient way to untie these two modules?
        # model.pretrain_model.decoder.embed_tokens = (model.pretrain_model.shared.weight.clone())
        model.pretrain_model.decoder.embed_tokens = copy.deepcopy(model.pretrain_model.encoder.embed_tokens)

    return model, tokenizer_fast, training_args, model_args, task_args


def load_raw_dataset(data_filepath, db_path, schema_cache=None):
    with open(data_filepath, encoding="utf-8") as f:
        spider = json.load(f)

    if schema_cache is None:
        schema_cache = load_raw_dataset.SCHEMA_CACHE

    out_dataset = []
    for idx, sample in enumerate(spider):
        db_id = sample["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = dump_db_json_schema(
                db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
            )
        schema = schema_cache[db_id]
        out_dataset.append({
            "query": sample["query"],
            "question": sample["question"],
            "db_id": db_id,
            "db_path": db_path,
            "db_table_names": schema["table_names_original"],
#             "db_column_names": [
#                 {"table_id": table_id, "column_name": column_name}
#                 for table_id, column_name in schema["column_names_original"]
#             ],
            "db_column_names": {
                "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                "column_name": [column_name for _, column_name in schema["column_names_original"]]
            },
            "db_column_types": schema["column_types"],
#             "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
            "db_primary_keys": {
                "column_id": schema["primary_keys"]
            },
#             "db_foreign_keys": [
#                 {"column_id": column_id, "other_column_id": other_column_id}
#                 for column_id, other_column_id in schema["foreign_keys"]
#             ],
            "db_foreign_keys": {
                "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]
            },
            "rat_sql_graph": sample["rat_sql_graph"]
        })
    return out_dataset

load_raw_dataset.SCHEMA_CACHE = dict()


def run_model_forward_uskg(
    model,
    input_ids,
    attention_mask,
    decoder_input_ids,
    decoder_attention_mask,
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
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        past_prompt=past_prompt
    )

    return model_out


def separate_punct(s, exclude='_'):
    """
    avg(age), min(age)  ==>  avg ( age ) , min ( age )
    exclude (Iterable): the puncts not to separate (e.g. "." as in "t1.age")
    by default: _ (song_name)
    include: . (t1.age -> t1 . age)   % ('%hey%' -> ' % hey % ')
    """
    sep_punct = ''.join([c for c in string.punctuation if c not in exclude])
    sep_pattern = fr'([{re.escape(sep_punct)}])'
    sep_s = re.sub(sep_pattern, r' \1 ', s)
    # remove extra spaces
    sep_s = ' '.join(sep_s.strip().split())
    return sep_s


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def decode_sentences(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_sentences(tokenizer, row) for row in token_array]
    return tokenizer.decode(token_array)


# YS TODO: change result to a full list, since for uskg we do not assume corruption range is continuous
def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


# YS: other ways to corrupt
def find_text_struct_in_range(tokenizer, token_array):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(USKG_SPLITTER_CHARS)
    loc = 0
    text_tok_end, struct_tok_start = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if text_tok_end is None and loc > char_loc:
            text_tok_end = i
        if struct_tok_start is None and loc >= char_loc + len(USKG_SPLITTER_CHARS):
            struct_tok_start = i + 1
            break
    return (0, text_tok_end), (struct_tok_start, len(toks) - 1)     # the last token is </s>, don't corrupt it


def parse_struct_in(struct_in):
    """ Return: db_id, [table_name, [col_name, [vals]]] , each name is a tuple as (word idx, name) """
    st_words = struct_in.split()        # struct_in might still contain multiple spaces

    _, _db_id, _table_cols = struct_in.split('|', 2)
    db_id = _db_id.strip()
    db_id_t = (1, db_id, db_id)

    w_idx = 3       # struct_in starts with "| db_id | ..."
    parsed_struct = []    # [(tab_name, [col_names])]

    def _index_name(name, full_str=None):
        nonlocal w_idx
        name = name.strip()
        full_str = (full_str or name).strip()

        name_t = (w_idx, name, full_str)
        l = len(name.split(' '))
        _recs_str = ' '.join(st_words[w_idx : w_idx + l])
        assert _recs_str == name, (w_idx, _recs_str, name)
        w_idx += l + 1       # +1 for splitter
        return name_t

    tables = _table_cols.strip().split('|')
    for t in tables:
        tab_name, cols_str = t.strip().split(' : ')
        tab_name_t = _index_name(tab_name)

        # the substring with multiple values contain "," that shouldn't be splitted for columns
        val_split_pattern = r'\([^\)]*?\s,\s.*?\)'
        val_split_ranges = []
        # replace these "," with "@"
        for m in re.finditer(val_split_pattern, cols_str):
            s, e = m.span()
            val_split_ranges.append((s, e))
            cols_str = cols_str[:s] + cols_str[s:e].replace(',', '@') + cols_str[e:]

        cols = cols_str.split(' , ')
        parsed_cols = []
        for c in cols:
            # No matter having value or not, use the full column string as column name range
            if "( " not in c:   # spaces to avoid "official_ratings_(millions)" error; same below
                # No values, just col name
                parsed_cols.append([_index_name(c), []])
                continue
            # Has values
            lb_char_idx = c.index("( ")         
            rb_char_idx = c.rindex(" )") + 1    # +1 for the preceding space
            col_name = c[:lb_char_idx].strip()
            col_name_t = _index_name(col_name, full_str=c)
            vals_str = c[lb_char_idx+1:rb_char_idx].strip()
            vals = vals_str.split(' @ ')    # previously replaced "," with "@"
            parsed_vals = []
            for v in vals:
                parsed_vals.append(_index_name(v))
            w_idx += 1                      # for ")"  (no need to +1 for "(" because there is by default one splitter; for ") ," there are two, so +1)
            parsed_cols.append([col_name_t, parsed_vals])

        parsed_struct.append( (tab_name_t, parsed_cols) )

    return db_id_t, parsed_struct


def find_struct_name_ranges(tokenizer, token_array, struct_in):
    """ 
    (Need to pass struct_in since it's easier to parse then the string reconstructed by tokenizer)

    YS TODO: allowing specifying the table name of the column to select a specific column from all with the same name
    """
    # sentence = decode_sentences(tokenizer, token_array)
    # assert USKG_SPLITTER in sentence, f'"{sentence}" does not have splitter {USKG_SPLITTER}'
    # # select the struct_in part
    text_range, struct_range = find_text_struct_in_range(tokenizer, token_array)
    sb, se = struct_range
    # token_array = token_array[sb : se]
    # struct_in = ' '.join(decode_tokens(tokenizer, token_array))

    parsed_struct_in = parse_struct_in(struct_in)
    db_id_t, tables = parsed_struct_in
    stk = tokenizer(struct_in)

    db_id_ranges = defaultdict(list)
    table_name_ranges = defaultdict(list)      # dict[name, [(st, ed)]]; same for others
    col_name_ranges = defaultdict(list)
    val_name_ranges = defaultdict(list)
    
    def _add_t(name_t, d, extra_name_t=None):
        # """
        # extra_name_t: for column, provide the value tuples to add their range to column
        #     (TODO: add a "table" key besides "table_name", which includes all tokens under the table)
        # """
        nonlocal stk
        nonlocal sb, se
        w_idx, name, full_str = name_t
        # name_words = name.split(' ')
        # w_idx_last = w_idx + len(name_words) - 1
        full_words = full_str.split()
        w_idx_last = w_idx + len(full_words) - 1
        s, _ = stk.word_to_tokens(w_idx)
        _, e = stk.word_to_tokens(w_idx_last)
        d[name].append((s + sb, e + sb))

    _add_t(db_id_t, db_id_ranges)
    for table_name_t, cols in tables:
        _add_t(table_name_t, table_name_ranges)
        for col_name_t, vals in cols:
            _add_t(col_name_t, col_name_ranges)
            for val_t in vals:
                _add_t(val_t, val_name_ranges)

    return dict(
        db_id_ranges=db_id_ranges,
        table_name_ranges=table_name_ranges,
        col_name_ranges=col_name_ranges,
        val_name_ranges=val_name_ranges,
    )

def make_dec_prompt(dec_target, subject):
    dec_target = ' ' + dec_target + ' '  # to avoid the matching problems at the ends 
    m = re.search(fr'\W({subject})\W', dec_target)
    assert m is not None
    s, e = m.span(1)
    prompt = dec_target[:s].strip()
    return prompt

def ensure_list(x):
    """ If x is singleton (int, float, etc.), make it a singleton-list """
    try:
        _ = x[0]
    except:
        # singleton
        x = [x]
    return x
