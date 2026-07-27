import re
import json

## for recovering train, val curves from the training logs if forgot to set metrics_uri for train phase.
# (1) cat train_output.log | grep NDCG@20 >& tmp_metrics.txt
# (2) python3 train_logs_parser

def parse_metrics(input_filepath, output_filepath):
    # Initialize the dictionary to match the target JSON structure
    metrics = {
        "train_loss": {"x": [], "y": []},
        "train_mrr_20": {"x": [], "y": []},
        "train_ndcg_20": {"x": [], "y": []},
        "train_recall_20": {"x": [], "y": []},
        "val_loss": {"x": [], "y": []},
        "val_mrr_20": {"x": [], "y": []},
        "val_ndcg_20": {"x": [], "y": []},
        "val_recall_20": {"x": [], "y": []}
    }

    # Regex pattern to capture the epoch and all 8 metric values
    # Note: Handles the missing space in "...recall_20 {val}avg val loss..."
    pattern = re.compile(
        r"Epoch (\d+): Train avg Loss ([\d.]+) \| "
        r"train NDCG@20 ([\d.]+) \| train MRR@20 ([\d.]+) \| "
        r"train recall_20 ([\d.]+)avg val loss ([\d.]+) \| "
        r"val NDCG@20 ([\d.]+) \| val MRR@20 ([\d.]+) \| "
        r"val recall_20 ([\d.]+)"
    )

    # Read the text file and parse line by line
    with open(input_filepath, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))

                # Append the epoch (x) and casted float metric (y) to their respective lists
                metrics["train_loss"]["x"].append(epoch)
                metrics["train_loss"]["y"].append(float(match.group(2)))

                metrics["train_ndcg_20"]["x"].append(epoch)
                metrics["train_ndcg_20"]["y"].append(float(match.group(3)))

                metrics["train_mrr_20"]["x"].append(epoch)
                metrics["train_mrr_20"]["y"].append(float(match.group(4)))

                metrics["train_recall_20"]["x"].append(epoch)
                metrics["train_recall_20"]["y"].append(float(match.group(5)))

                metrics["val_loss"]["x"].append(epoch)
                metrics["val_loss"]["y"].append(float(match.group(6)))

                metrics["val_ndcg_20"]["x"].append(epoch)
                metrics["val_ndcg_20"]["y"].append(float(match.group(7)))

                metrics["val_mrr_20"]["x"].append(epoch)
                metrics["val_mrr_20"]["y"].append(float(match.group(8)))

                metrics["val_recall_20"]["x"].append(epoch)
                metrics["val_recall_20"]["y"].append(float(match.group(9)))

    # Export the populated dictionary to a JSON file
    with open(output_filepath, 'w') as json_file:
        json.dump(metrics, json_file)
        print(f"Metrics successfully parsed and saved to {output_filepath}")

if __name__ == "__main__":
    # Ensure tt_metrics.txt is in the same directory as this script
    parse_metrics("tmp_metrics.txt", "metrics.json")