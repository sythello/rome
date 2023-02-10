import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
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


def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="t5-base",
    )
    # aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            plot_trace_heatmap(plot_result, savepdf=pdfname)


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


def trace_with_patch_t5(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end) (encoder input)
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

    (YS) For now, assuming only corrupting encoder input sentence
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername_t5(model, "encoder", 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        # x: original layer output; Return: maybe modified layer output
        # x: (bsize, tokens, ...)
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
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
        # (YS) TODO: automatically decide how to forward
        # outputs_exp = model(**inp)
        outputs_exp = run_model_forward_uskg(model, **inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_with_repatch_t5(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
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

    embed_layername = layername_t5(model, "encoder", 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            # (YS) TODO: automatically decide how to forward
            # outputs_exp = model(**inp)
            outputs_exp = run_model_forward_uskg(model, **inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow_t5(
    mt,
    enc_sentence,
    dec_prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs_t5(mt.tokenizer, [enc_sentence] * (samples + 1), [dec_prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)

    # TODO: can corrupt tokens other than "subjects"
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")

    low_score = trace_with_patch_t5(
        mt.model,
        inp=inp,
        states_to_patch=[], 
        answers_t=answer_t, 
        tokens_to_mix=e_range, 
        noise=noise, 
        uniform_noise=uniform_noise,
    ).item()
    if not kind:
        differences = trace_important_states_t5(
            mt.model,
            mt.num_enc_layers,
            mt.num_dec_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window_t5(
            mt.model,
            mt.num_enc_layers,
            mt.num_dec_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    # differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        dec_input_ids=inp["decoder_input_ids"][0],
        dec_input_tokens=decode_tokens(mt.tokenizer, inp["decoder_input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states_t5(
    model,
    num_enc_layers,
    num_dec_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    """ 
    e_range: the tokens to corrupt (for now, it's only for encoder input. TODO: add another argument for decoder input corruption)
    token_range: the indexes of tokens to try to restore (by default all tokens)
    """
    enc_ntoks = inp["input_ids"].shape[1]
    dec_ntoks = inp["decoder_input_ids"].shape[1]

    # table: (ntoks, nlayers)
    enc_table = []
    dec_table = []

    # TODO: split `token_range` into `enc_token_range` and `dec_token_range`
    if token_range is not None:
        print('* Warning: token_range not implemented for trace_important_states_t5(), ignored for now')

    # (YS): Encoder part
    enc_tqdm_pbar = tqdm(total=enc_ntoks*num_enc_layers,
                         desc="trace_important_states_t5.encoder")
    enc_token_range = range(enc_ntoks)
    # if token_range is None:
    #     token_range = range(ntoks)
    for tnum in enc_token_range:
        row = []
        for layer in range(num_enc_layers):
            r = trace_with_patch_t5(
                model,
                inp=inp,
                states_to_patch=[(tnum, layername_t5(model, 'encoder', layer))],
                answers_t=answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
            enc_tqdm_pbar.update(1)
        enc_table.append(torch.stack(row))
    enc_table_pt = torch.stack(enc_table).detach().cpu()
    enc_tqdm_pbar.close()

    # (YS): Decoder part
    dec_tqdm_pbar = tqdm(total=dec_ntoks*num_dec_layers,
                         desc="trace_important_states_t5.decoder")
    dec_token_range = range(dec_ntoks)
    # if token_range is None:
    #     token_range = range(ntoks)
    for tnum in dec_token_range:
        row = []
        for layer in range(num_dec_layers):
            r = trace_with_patch_t5(
                model,
                inp=inp,
                states_to_patch=[(tnum, layername_t5(model, 'decoder', layer))],
                answers_t=answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
            dec_tqdm_pbar.update(1)
        dec_table.append(torch.stack(row))
    dec_table_pt = torch.stack(dec_table).detach().cpu()
    dec_tqdm_pbar.close()

    return enc_table_pt, dec_table_pt


def trace_important_window_t5(
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
    token_range=None,
):
    enc_ntoks = inp["input_ids"].shape[1]
    dec_ntoks = inp["decoder_input_ids"].shape[1]

    # table: (ntoks, nlayers)
    enc_table = []
    dec_table = []

    # TODO: split `token_range` into `enc_token_range` and `dec_token_range`
    if token_range is not None:
        print('* Warning: token_range not implemented for trace_important_window_t5(), ignored for now')

    # (YS): Encoder part
    enc_tqdm_pbar = tqdm(total=enc_ntoks*num_enc_layers,
                         desc="trace_important_window_t5.encoder")
    enc_token_range = range(enc_ntoks)
    for tnum in enc_token_range:
        row = []
        for layer in range(num_enc_layers):
            layerlist = [
                (tnum, layername_t5(model, 'encoder', L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_enc_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch_t5(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
            enc_tqdm_pbar.update(1)
        enc_table.append(torch.stack(row))
    enc_table_pt = torch.stack(enc_table).detach().cpu()
    enc_tqdm_pbar.close()

    # (YS): Decoder part
    dec_tqdm_pbar = tqdm(total=dec_ntoks*num_dec_layers,
                         desc="trace_important_window_t5.decoder")
    dec_token_range = range(dec_ntoks)
    for tnum in dec_token_range:
        row = []
        for layer in range(num_dec_layers):
            layerlist = [
                (tnum, layername_t5(model, 'decoder', L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_dec_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch_t5(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
            dec_tqdm_pbar.update(1)
        dec_table.append(torch.stack(row))
    dec_table_pt = torch.stack(dec_table).detach().cpu()
    dec_tqdm_pbar.close()

    return enc_table_pt, dec_table_pt


class ModelAndTokenizer_T5:
    """
    An object to hold on to (or automatically download and hold)
    a T5-style tokenizer.  Counts the number of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()

        self.tokenizer = tokenizer
        self.model = model
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
            if (re.match(r"^encoder\.block\.\d+$", n))
        ]
        self.dec_layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^decoder\.block\.\d+$", n))
        ]
        self.layer_names = self.enc_layer_names + self.dec_layer_names

        self.num_enc_layers = len(self.enc_layer_names)
        self.num_dec_layers = len(self.dec_layer_names)
        self.num_layers = len(self.layer_names)
        

    def __repr__(self):
        return (
            f"ModelAndTokenizer_T5(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername_t5(model, part, num, kind=None):
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
    assert hasattr(model, part), f"{part} not in the given model of type {type(model)}"

    # Layer names
    # decoder.block.2
    # decoder.block.2.layer.0.SelfAttention
    # decoder.block.2.layer.1.EncDecAttention
    # decoder.block.2.layer.2.DenseReluDense

    if kind == "embed":
        return f"{part}.embed_tokens"

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
    
    return f"{part}.block.{num}{'' if _kind is None else '.' + _kind}"


# def guess_subject(prompt):
#     return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
#         0
#     ].strip()


def plot_hidden_flow_t5(
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
    if subject is None:
        # subject = guess_subject(prompt)
        raise NotImplementedError('TODO: add automatic finding subject of different types (column, table, op, ...)')
    result = calculate_hidden_flow_t5(
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
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"
    dec_labels = list(result["dec_input_tokens"])

    def _draw_single_plot(fig, ax, differences, labels, modelname=modelname, title=title, part="all", kind=kind, answer=answer):
        h = ax0.pcolor(
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
            modelname = "T5"
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

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(3.5, 5), dpi=200)
        _draw_single_plot(fig, ax0, enc_differences, labels, part="encoder")
        _draw_single_plot(fig, ax1, dec_differences, dec_labels, part="decoder")
        fig.tight_layout()
        
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow_t5(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow_t5(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs_t5(tokenizer, enc_sentences, dec_prompts, device="cuda"):
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


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


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


def predict_token_t5(mt, enc_sentences, dec_prompts, return_p=False):
    inp = make_inputs_t5(mt.tokenizer, enc_sentences, dec_prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    # (YS) TODO: automatically decide how to forward
    # out = model(**inp)
    out = run_model_forward_uskg(model, **inp)
    # print(out.keys())
    out = out["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs_t5(mt.tokenizer, [s], [])
        with nethook.Trace(mt.model, layername_t5(mt.model, "encoder", 0, "embed")) as t:
            mt.model(**inp)
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
