import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import shutil
import os

FOLDER_PATH = "results/llama-2-7b-80k/"
MODEL_NAME = "LLaMA 2 7B continue-trained on 5B tokens 80K length Per-source length upsampled data"
PRETRAINED_LEN = 81920


def run_vis(args=None):
    if args is None:
        main()
    else:
        args_main(args)


def args_main(args, model_name_suffix=None):
    # Path to the directory containing JSON results

    if (args.model_path is not None):
        assert (args.model_name is None)
        model_name = args.model_path
    else:
        assert (args.model_name is not None)
        model_name = args.model_name

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    else:
        model_name = model_name

    original_model_name = model_name

    if model_name_suffix is not None:
        model_name += "_" + model_name_suffix

    folder_path = os.path.join(args.log_dir, f"results/{model_name}")
    # print(args.log_dir, model_name)
    # folder_path = os.path.join(args.log_dir, model_name)
    print(f"load results from {folder_path}")
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/*.json")
    # import ipdb; ipdb.set_trace()
    print(f"file nums: {len(list(json_files))}")
    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            # [5-31 yuzhaouoe] fix the metric
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            # score = len(set(model_response.split()).intersection(set(expected_answer))) / len(expected_answer)
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    max_len_to_plot = 40 * 1024
    data = [item for item in data if item["Context Length"] <= max_len_to_plot]

    # Creating a DataFrame
    df = pd.DataFrame(data)
    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if (l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    overall_score = df["Score"].mean()
    print("Overall score %.3f" % overall_score)

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'],
                                 aggfunc='mean').reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length",
                                    values="Score")  # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )

    # More aesthetics
    # model_name_ = MODEL_NAME
    plt.title(
        f'Pressure Testing {original_model_name}, skip layers: {args.skip_layers}, keep ratio: {args.keep_ratio}. '
        f'Overall score: {overall_score:.3f}\n'
        f'Fact Retrieval Across Context Lengths ("Needle In A HayStack")'
    )  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    # plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    # save_path = os.path.join(args.log_dir, f"img/{model_name}.png")
    save_path = f"./output/images/{model_name}.pdf"

    # save_path = "img/%s.png" % model_name
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)


def main():
    # Path to the directory containing JSON results
    folder_path = FOLDER_PATH
    if ("/" in folder_path):
        model_name = folder_path.split("/")[-2]
    else:
        model_name = MODEL_NAME
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            # [5-31 yuzhaouoe] fix the metric
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            # score = len(set(model_response.split()).intersection(set(expected_answer))) / len(expected_answer)
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if (l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'],
                                 aggfunc='mean').reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length",
                                    values="Score")  # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )

    # More aesthetics
    model_name_ = MODEL_NAME
    plt.title(
        f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    save_path = "img/%s.png" % model_name
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    run_vis(None)
