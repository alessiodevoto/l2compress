import argparse
from transformers import LlamaForCausalLM, LlamaModel
from gen_utils import get_adjust_kv_strategy
from needle.needle_in_haystack import LLMNeedleHaystackTester
from needle.needle_visualize import args_main
from needle.needle_utils import get_suffix
import logging

from utils import cast_to_hidden_state_drop

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number',
                        default=0)
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number',
                        default=128000)
    parser.add_argument('--model_provider', type=str, help='which model to use',
                        default="LLaMA")
    parser.add_argument('--model_path', type=str, help='path to model',
                        default="./models/llama-2-7b-80k")
    parser.add_argument('--log_dir', type=str, default="./needle")
    parser.add_argument('--more_suffix', type=str, default=None)
    parser.add_argument('--hidden_drop', action="store_true")

    # do not use the following args
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')

    # parser = add_args(parser)

    # kv cache compression settings
    parser.add_argument("--sort_by", type=str, default="key", help="key or value")
    parser.add_argument("--sort_metric", type=str, default="norm", help="norm, kurtosis or random")
    # parser.add_argument("--sort_descending", type=bool, default=False, help="sort in descending order, i.e. keep the largest norms or kurtosis")
    parser.add_argument("--sort_descending", action="store_true",
                        help="sort in descending order, i.e. keep the largest norms or kurtosis")
    parser.add_argument("--keep_ratio", type=float, default=1)
    parser.add_argument("--prune_after", type=int, default=0)
    parser.add_argument("--skip_layers", type=str, default="0", help="comma separated list of layers to skip")

    parser.add_argument("--snapkv", action="store_true", help="snapkv")
    parser.add_argument("--do_not_save_results", action="store_true")
    parser.add_argument("--only_visualise", action="store_true")
    args = parser.parse_args()

    if args.snapkv:
        from snapkv.monkeypatch.monkeypatch import replace_llama
        replace_llama()  # Use monkey patches enable SnapKV
        logger.info("replace llama with SnapKV patch")
    else:
        if "llama-2-7b-80k" in args.model_path:
            from needle.replace_attention import replace_hf35

            replace_hf35()

    if args.model_path is not None:
        assert args.model_name is None
        model_name = args.model_path
    else:
        assert args.model_name is not None
        model_name = args.model_name

    # if not args.hidden_drop:
    adjust_kv_strategy = get_adjust_kv_strategy(
        skip_layers=args.skip_layers,
        keep_ratio=args.keep_ratio,
        sort_by=args.sort_by,
        prune_after=args.prune_after,
        sort_metric=args.sort_metric,
        sort_descending=args.sort_descending,
    )
    logger.info(f"adjust_kv_strategy: {adjust_kv_strategy}")
    # else:
    #     adjust_kv_strategy = None

    if args.snapkv:
        suffix = "snapkv"
        if args.more_suffix is not None:
            suffix = suffix + f"_{args.more_suffix}"
    else:
        suffix = get_suffix(args.skip_layers, args.keep_ratio, args.sort_metric,
                            args.hidden_drop, args.more_suffix)

    if not args.only_visualise:
        # try:
        ht = LLMNeedleHaystackTester(
            haystack_dir="./needle/PaulGrahamEssays",
            log_dir=args.log_dir,
            adjust_kv_strategy=adjust_kv_strategy,
            model_name=model_name,
            model_provider=args.model_provider,
            save_contexts=not args.do_not_save_results,
            save_results=not args.do_not_save_results,
            openai_api_key=args.api_key,
            model_name_suffix=suffix,
            snapkv=args.snapkv,
        )

        if args.hidden_drop:
            if isinstance(args.skip_layers, str):
                layers_to_skip = [int(l) for l in args.skip_layers.split(',')]
            else:
                assert isinstance(args.skip_layers, list)
                layers_to_skip = args.skip_layers

            ht.model_to_test = cast_to_hidden_state_drop(
                ht.model_to_test,
                # keep_ratio=0.6,
                # skip_layers=list(range(32))[:-10],
                keep_ratio=args.keep_ratio,
                skip_layers=layers_to_skip,
                sort_metric='norm',
                sort_descending=True,
            )
            logger.info("cast hidden state drop decoder layers.")

        ht.start_test(args)
        # except Exception as e:
        #     logger.error(e)
    else:
        args_main(args, model_name_suffix=suffix)


if __name__ == '__main__':
    # try:
    args = main()
    # except Exception as e:
    #     print(e)

    # print("exit")
    # exit(0)
