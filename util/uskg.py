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
from uskg.third_party.spider import evaluation as sp_eval


USKG_SPLITTER = '; structed knowledge: '
USKG_SPLITTER_CHARS = ';structedknowledge:'

RAT_SQL_RELATION_ID2NAME = {
    0: ('qq_dist', -2),
    1: ('qq_dist', -1),
    2: ('qq_dist', 0),
    3: ('qq_dist', 1),
    4: ('qq_dist', 2),
    5: 'qc_default',
    6: 'qt_default',
    7: 'cq_default',
    8: 'cc_default',
    9: 'cc_foreign_key_forward',
    10: 'cc_foreign_key_backward',
    11: 'cc_table_match',
    12: ('cc_dist', -2),
    13: ('cc_dist', -1),
    14: ('cc_dist', 0),
    15: ('cc_dist', 1),
    16: ('cc_dist', 2),
    17: 'ct_default',
    18: 'ct_foreign_key',
    19: 'ct_primary_key',
    20: 'ct_table_match',
    21: 'ct_any_table',
    22: 'tq_default',
    23: 'tc_default',
    24: 'tc_primary_key',
    25: 'tc_table_match',
    26: 'tc_any_table',
    27: 'tc_foreign_key',
    28: 'tt_default',
    29: 'tt_foreign_key_forward',
    30: 'tt_foreign_key_backward',
    31: 'tt_foreign_key_both',
    32: ('tt_dist', -2),
    33: ('tt_dist', -1),
    34: ('tt_dist', 0),
    35: ('tt_dist', 1),
    36: ('tt_dist', 2),
    37: 'qcCEM',
    38: 'cqCEM',
    39: 'qtTEM',
    40: 'tqTEM',
    41: 'qcCPM',
    42: 'cqCPM',
    43: 'qtTPM',
    44: 'tqTPM',
    45: 'qcNUMBER',
    46: 'cqNUMBER',
    47: 'qcTIME',
    48: 'cqTIME',
    49: 'qcCELLMATCH',
    50: 'cqCELLMATCH'
}

REL2ID = {v : k for k, v in RAT_SQL_RELATION_ID2NAME.items()}


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
    print('Using tokenizer_uskg:', model_path)
    tokenizer_uskg = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print('Using tokenizer_fast:', model_args.bert.location)
    tokenizer_fast = AutoTokenizer.from_pretrained(model_args.bert.location, use_fast=True)

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

    return model, tokenizer_uskg, tokenizer_fast, training_args, model_args, task_args


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


def load_spider_dataset(args, mt):
    raw_spider_dataset = load_raw_dataset(
        data_filepath = args.spider_dataset_path,
        db_path=args.spider_db_dir,
    )
    if args.ds == 'train':
        dataset_cls = s2s_spider.TrainDataset
    else:
        dataset_cls = s2s_spider.DevDataset
    processed_spider_dataset = dataset_cls(
        args=mt.task_args,
        raw_datasets=raw_spider_dataset,
        cache_root=args.data_cache_dir
    )
    return processed_spider_dataset


def run_model_forward_uskg(
    model,
    input_ids,
    attention_mask,
    decoder_input_ids,
    decoder_attention_mask,
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
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_attentions=output_attentions,
        past_prompt=past_prompt
    )

    return model_out


class ModelAndTokenizer_USKG:
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
        model, tokenizer_uskg, tokenizer_fast, training_args, model_args, task_args = load_model_uskg(model_name, untie_embeddings=True)
        model = model.eval().to(device=device)

        self.tokenizer = tokenizer_fast
        self.tokenizer_uskg = tokenizer_uskg
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


def load_evaluator(args):
    table_path = args.spider_tables_path
    db_dir = args.spider_db_dir

    kmaps = sp_eval.build_foreign_key_map_from_json(table_path)
    evaluator = sp_eval.Evaluator(db_dir=db_dir, kmaps=kmaps, etype='all')
    
    return evaluator

# def guess_subject(prompt):
#     return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
#         0
#     ].strip()


def separate_punct(s, exclude='_'):
    """
    avg(age), min(age)  ==>  avg ( age ) , min ( age )
    exclude (Iterable): the puncts not to separate (e.g. "_" as in "pet_type")
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
            # Must be ' , ' instead of ',' because there are values with ',' such as "Louisville, Kentucky" in shop_membership
            cols_str = cols_str[:s] + cols_str[s:e].replace(' , ', ' @ ') + cols_str[e:]

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


# def find_struct_name_ranges(tokenizer, token_array, struct_in):
def find_struct_name_ranges(tokenizer, ex):
    """ 
    Need ex keys:
        struct_in, enc_sentence
    (Need to pass struct_in since it's easier to parse than the string reconstructed by tokenizer)

    YS TODO: allowing specifying the table name of the column to select a specific column from all with the same name
    """

    text_in = ex['text_in']
    struct_in = ex['struct_in']
    if 'enc_sentence' not in ex:
        ex['enc_sentence'] = f"{text_in}; structed knowledge: {struct_in}"

    token_array = tokenizer(ex['enc_sentence'])['input_ids']
    

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

    struct_node_ranges_dict = dict(
        db_id_ranges=db_id_ranges,
        table_name_ranges=table_name_ranges,
        col_name_ranges=col_name_ranges,
        val_name_ranges=val_name_ranges,
    )

    ex['text_range'] = text_range
    ex['struct_range'] = struct_range
    ex['struct_node_ranges_dict'] = struct_node_ranges_dict

    return struct_node_ranges_dict


def make_dec_prompt(dec_target, subject):
    dec_target = ' ' + dec_target + ' '  # to avoid the matching problems at the ends 
    # m = re.search(fr'\W({subject})\W', dec_target)
    m_iter = re.finditer(fr'\W({subject})\W', dec_target)
    prompts = []
    for m in m_iter:
        s, e = m.span(1)
        prompt = dec_target[:s].strip()
        prompts.append(prompt)

    assert prompts, (dec_target, subject)
    
    return prompts

def ensure_list(x):
    """ If x is singleton (int, float, etc.), make it a singleton-list """
    """ TODO: make this safer (e.g. for str, it will pass the test as a list of chars) """
    try:
        _ = list(x)
    except:
        # singleton
        x = [x]
    return x

def evaluate_hardness(sql_str, db_id, args=None, evaluator=None):
    if evaluator is None:
        try:
            # use saved evaluator
            evaluator = evaluate_hardness.evaluator
        except AttributeError:
            # create and save a new evaluator
            assert args is not None
            evaluate_hardness.evaluator = load_evaluator(args)
    else:
        # evaluator is not None; save the passed-in evaluator
        evaluate_hardness.evaluator = evaluator

    schema = evaluator.schemas[db_id]
    _sql = sp_eval.get_sql(schema, sql_str)
    hardness = evaluator.eval_hardness(_sql)
    return hardness

# evaluate_hardness.evaluator = None


# def detect_column_role(dec_prompt):
#     role_keyword_pattern = r'\W(select|where|join|group by|having|order by)\W'
#     all_kws = re.findall(role_keyword_pattern, ' ' + dec_prompt + ' ')
#     assert len(all_kws) > 0, dec_prompt
#     col_role_kw = all_kws[-1]
#     return col_role_kw


def detect_node_role(dec_prompt):
    # didn't include "on" and "as" to put all "join" results together 
    role_keyword_pattern = r'\W(select|from|where|join|group by|having|order by)\W'
    all_kws = re.findall(role_keyword_pattern, ' ' + dec_prompt + ' ')
    assert len(all_kws) > 0, dec_prompt
    role_kw = all_kws[-1]
    return role_kw    


def check_col_text_match(spider_ex, col, tab=None):
    # col = result_d['expect']
    # tab = result_d['table']
    # node_name = f'<C>{tab}::{col}'
    nodes = spider_ex['rat_sql_graph']['nodes']
    
    """ TODO: if tab not given, infer from parsed_struct_in (first appearing col name) """
    if tab is None:
        raise NotImplementedError
    
    struct_in = spider_ex['struct_in']
    
    # use struct_in to find column id in table and table id; use these to index ratsql node
    db_id, tables = parse_struct_in(spider_ex['struct_in'])
    for tid, (tab_name_t, cols) in enumerate(tables):
        if tab_name_t[1] != tab:
            continue
        for cid, (col_name_t, vals) in enumerate(cols):
            if col_name_t[1] != col:
                continue
            # Found the node table/column; save them 
            node_tid = tid
            node_cid = cid
            break
            
    all_tab_nodes = [n for n in nodes if n.startswith('<T>')]
    tab_node = all_tab_nodes[node_tid]
    ratsql_tab_name = tab_node.split('<T>')[1]
    tab_prefix = f'<C>{ratsql_tab_name}::'
    tab_all_col_nodes = [n for n in nodes if n.startswith(tab_prefix)]
    col_node = tab_all_col_nodes[node_cid]
    node_idx = nodes.index(col_node)
    
    rel_matrix = json.loads(spider_ex['rat_sql_graph']['relations'])
    rel_row = rel_matrix[node_idx]
    
    if REL2ID['cqCEM'] in rel_row:
        return 'exact'
    elif REL2ID['cqCPM'] in rel_row:
        return 'partial'
    else:
        return 'no-match'


def check_table_text_match(spider_ex, tab):
    nodes = spider_ex['rat_sql_graph']['nodes']

    # use struct_in to find table id; use these to index ratsql node
    # (cannot directly index ratsql nodes because there they do lemmatization)
    db_id, tables = parse_struct_in(spider_ex['struct_in'])
    for tid, (tab_name_t, cols) in enumerate(tables):
        if tab_name_t[1] == tab:
            node_tid = tid
            break

    all_tab_nodes = [n for n in nodes if n.startswith('<T>')]
    tab_node = all_tab_nodes[node_tid]
    node_idx = nodes.index(tab_node)
    
    rel_matrix = json.loads(spider_ex['rat_sql_graph']['relations'])
    rel_row = rel_matrix[node_idx]
    
    if REL2ID['tqTEM'] in rel_row:
        return 'exact'
    elif REL2ID['tqTPM'] in rel_row:
        return 'partial'
    else:
        return 'no-match'


def parse_sql_alias2table(sql_str):
    """ Assume that alias-table binding only happens in format like '... JOIN table AS t1 ...' """
    sql_tokens = sql_str.lower().strip().split()
    alias2table = dict()

    for i, t in enumerate(sql_tokens):
        if t == 'as':
            assert 0 < i < len(sql_tokens) - 1, sql_str
            assert sql_tokens[i-2] in {'from', 'join'}, sql_str
            table_name = sql_tokens[i-1]
            alias = sql_tokens[i+1]
            assert alias.startswith('t'), alias
            alias2table[alias] = table_name

    return alias2table

def nested_list_processing(nl, func):
    """
    nl: a nested list
    func: a Callable to process each item in `nl`
    """
    if isinstance(nl, list):
        processed_nl = [nested_list_processing(l, func) for l in nl]
        return processed_nl
    else:
        return func(nl)

def nested_json_processing(obj, func):
    """
    obj: a nested json object (with lists and dicts)
    func: a Callable to process each "leaf" item in `obj`
    """
    if isinstance(obj, list):
        processed_obj = [nested_json_processing(elem, func) for elem in obj]
        return processed_obj
    elif isinstance(obj, dict):
        processed_obj = {k : nested_json_processing(v, func) for k, v in obj.items()}
        return processed_obj
    else:
        return func(obj)

# def plot_uskg_enc_attention(d, savepdf=None):
#     ## Assume 16 heads, 24 layers (T5 large config)
    
#     ## encoder self attention 
#     inspect_layers = [0, 6, 12, 18, 23]
#     att_dict = d['attentions']
    
#     cand_len = len(att_dict['enc_cand_tokens'])
#     head_len = len(att_dict['enc_head_tokens'])

#     fig_w = 22
#     fig_h = (0.11*cand_len + 1) * head_len
#     fig, ax_list = plt.subplots(
#         nrows=head_len,
#         ncols=len(inspect_layers),
#         squeeze=False,
#         figsize=(fig_w, fig_h))

#     att_mat = nested_list_processing(att_dict['enc_attn'], func=float)
#     att_mat = np.array(att_mat)
    
#     for expect_i in range(len(att_dict['enc_head_tokens'])):
#         for l_id, layer in enumerate(inspect_layers):
#             val_mat = att_mat[layer, :, expect_i, :]  # layer, all heads, expect tok i -> all toks 
#             val_mat = val_mat.transpose()    # (cand_toks, n_heads)
#             x_labels = range(val_mat.shape[1])
#             y_labels = att_dict['enc_cand_tokens']
#             title_toks = att_dict['enc_head_tokens'][:expect_i] + [f"*{att_dict['enc_head_tokens'][expect_i]}*"]
#             title = f"L{layer}  Head token: {' '.join(title_toks)}\n"
            
#             ax = ax_list[expect_i, l_id]
#             _draw_single_plot_2(ax,
#                                 val_mat=val_mat, 
#                                 x_labels=x_labels, 
#                                 y_labels=y_labels,
#                                 title=title)
            
#     fig.tight_layout()
#     if savepdf:
#         plt.savefig(savepdf, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()






