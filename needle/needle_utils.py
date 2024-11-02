import glob
import os


def load_context(fpath="eval/needle/PaulGrahamEssays/*.txt", ctx_len=100000):
    context = ""
    for file in glob.glob(fpath):
        with open(file, 'r') as f:
            context += f.read()
    LLAMA_CHAR_TO_TOKEN_RATIO = 3.66
    context = context[: int(ctx_len * LLAMA_CHAR_TO_TOKEN_RATIO)]
    return context


def insert_needle(context, needle, depth):
    context = context.split(".")
    c_len = len(context)
    needle_place = int(depth * c_len)
    context = ".".join(context[:needle_place]) + "." + needle + ".".join(context[needle_place:])
    return context


def get_suffix(skip_layers, keep_ratio, sort_metric, hidden_drop, more_suffix):
    if isinstance(skip_layers, list):
        skip_layers = "-".join(skip_layers)
    else:
        assert isinstance(skip_layers, str)
        skip_layers = skip_layers.replace(",", "-")
    keep_ratio = str(int(keep_ratio * 100))
    suffix = f"sort{sort_metric}_skip{skip_layers}_keep{keep_ratio}"
    # suffix = f"skip{skip_layers}_keep{keep_ratio}"
    if hidden_drop is True:
        suffix += f"_hiddendrop"
    if more_suffix is not None:
        suffix = suffix + "_" + more_suffix
    return suffix


def convert_model_id_to_name(model_name):
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    return model_name
