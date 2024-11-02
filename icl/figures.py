import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from pathlib import Path

sns.set_style("whitegrid")


def load_log_result(log_dir, icl_type):
    log_file = Path("../output/icl/") / log_dir / "results.json"
    records = json.load(open(log_file, "r", encoding="utf-8"))
    for idx in range(len(records)):
        records[idx].update({"ICL type": icl_type})
        if records[idx]["score"] < 0.001:
            records[idx]["score"] = 0.1
    records = [rec for rec in records if rec["shot"] != 0]
    return records


def main():
    plt.figure(figsize=(6, 5), dpi=180)
    records = []
    log_dir = "05_30_2024-13_29_09"
    records = records + load_log_result(log_dir, icl_type="flipped")
    log_dir = "05_30_2024-16_29_55"
    records = records + load_log_result(log_dir, icl_type="regular")
    log_dir = "05_30_2024-17_42_43"
    records = records + load_log_result(log_dir, icl_type="unrelated")
    log_dir = "05_31_2024-10_12_10"
    # records = records + load_log_result(log_dir, icl_type="flipped-jsonline")
    log_dir = "05_31_2024-12_35_53"
    # records = records + load_log_result(log_dir, icl_type="regular-jsonline")
    log_dir = "05_31_2024-14_41_28"
    # records = records + load_log_result(log_dir, icl_type="unrelated-jsonline")
    # yaofu/llama-2-7b-80k
    # log_dir = "06_04_2024-10_26_17"  # wrong Position Matters. ICL can learn position?
    # records = records + load_log_result(log_dir, icl_type="regular wrong llama2-7b-80k")
    # log_dir = "06_04_2024-15_42_05"  # wrong Position Matters. ICL can learn position?
    # records = records + load_log_result(log_dir, icl_type="unrelated wrong llama2-7b-80k")

    log_dir = "06_05_2024-03_10_09"
    records = records + load_log_result(log_dir, icl_type="regular llama2-7b-80k")
    log_dir = "06_05_2024-08_17_13"
    records = records + load_log_result(log_dir, icl_type="unrelated llama2-7b-80k")
    log_dir = "06_05_2024-13_23_27"
    records = records + load_log_result(log_dir, icl_type="flipped llama2-7b-80k")

    log_dir = "06_06_2024-01_50_02"
    records = records + load_log_result(log_dir, icl_type="regular-jsonline")
    log_dir = "06_06_2024-08_33_48"
    records = records + load_log_result(log_dir, icl_type="unrelated-jsonline")
    log_dir = "06_06_2024-15_05_08"
    records = records + load_log_result(log_dir, icl_type="flipped-jsonline")

    dataframe = pd.DataFrame.from_records(records)
    sns.lineplot(data=dataframe, x="shot", y="score", hue="ICL type",
                 markers=True, style="ICL type", dashes=False)
    plt.axhline(y=50, linestyle="--", color="grey", label="random", linewidth=1, alpha=0.7)
    plt.xscale("log")
    # plt.xlim(left=0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
