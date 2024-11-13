import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys
import numpy as np

import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader

from src import metrics, commons
from src.models import models
from src import frontends
from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.in_the_wild_dataset import InTheWildDataset
from src.datasets.WatermelonDataset import WatermelonDataset

import shap  # For SHAP feature importance
from clearml import Task, OutputModel

task = Task.init(project_name="Thesis_ADD", task_name="Model Evaluation Pure Results")

# Upload model
model_path = "/trained_models/specrnet_balanced/ckpt.pth"
output_model = OutputModel(task=task)
output_model.update_weights(weights_filename=model_path)


def get_dataset(datasets_paths: List[Union[Path, str]], amount_to_use: Optional[int]) -> SimpleAudioFakeDataset:
    data_val = WatermelonDataset(subset="val", path=datasets_paths[0])
    logging.info(f"Loaded dataset with {len(data_val)} samples.")
    return data_val


def evaluate_nn(model_paths: List[Path], datasets_paths: List[Union[Path, str]], model_config: Dict, device: str,
                amount_to_use: Optional[int] = None, batch_size: int = 16, output_dir="./visualizations"):
    logging.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]

    model = models.get_model(model_name=model_name, config=model_parameters, device=device)
    if len(model_paths):
        model.load_state_dict(torch.load(model_paths), strict=False)  # Allow partial loading if architecture differs
    model = model.to(device)

    data_val = get_dataset(datasets_paths=datasets_paths, amount_to_use=None)
    logging.info(f"Testing '{model_name}' model, weights path: '{model_paths}', on {len(data_val)} audio files.")
    test_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=3)

    num_correct = 0.0
    num_total = 0.0

    y_pred = torch.Tensor([]).to(device)
    y = torch.Tensor([]).to(device)
    y_pred_label = torch.Tensor([]).to(device)

    for i, (batch_x, _, batch_y) in enumerate(test_loader):
        model.eval()
        if i % 10 == 0:
            logging.info(f"Batch [{i}/{len(test_loader)}]")

        with torch.no_grad():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            num_total += batch_x.size(0)

            batch_pred = model(batch_x).squeeze(1)
            batch_pred = torch.sigmoid(batch_pred)
            batch_pred_label = (batch_pred + 0.5).int()

            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            y_pred = torch.concat([y_pred, batch_pred], dim=0)
            y_pred_label = torch.concat([y_pred_label, batch_pred_label], dim=0)
            y = torch.concat([y, batch_y], dim=0)

    # Compute SHAP feature importance on validation dataset
    visualize_shap_feature_importance(model, data_val, output_dir)

    eval_accuracy = (num_correct / num_total) * 100
    precision, recall, f1_score_final, support = precision_recall_fscore_support(
        y.cpu().numpy(), y_pred_label.cpu().numpy(), average="binary", beta=1.0
    )
    auc_score = roc_auc_score(y_true=y.cpu().numpy(), y_score=y_pred.cpu().numpy())

    logging.info(
        f"Accuracy: {eval_accuracy:.4f}%, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1 Score: {f1_score_final:.4f}, AUC: {auc_score:.4f}"
    )

    return y.cpu().numpy(), y_pred_label.cpu().numpy(), y_pred.cpu().numpy()


def visualize_metrics(y_true, y_pred_label, y_pred_prob, output_dir="./visualizations"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_true, y_pred_prob)
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_pred_prob)
    plt.title('ROC Curve')
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    # F1 Score at Different Thresholds
    thresholds = np.arange(0.0, 1.05, 0.05)
    f1_scores = [f1_score(y_true, y_pred_prob >= t) for t in thresholds]
    plt.plot(thresholds, f1_scores, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score at Different Thresholds")
    plt.savefig(f"{output_dir}/f1_score_thresholds.png")
    plt.close()

    # Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.savefig(f"{output_dir}/calibration_curve.png")
    plt.close()


def main(args):
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    commons.set_seed(seed)

    # Evaluate the model
    y_true, y_pred_label, y_pred_prob = evaluate_nn(
        model_paths=config["checkpoint"].get("path", []),
        datasets_paths=[args.in_the_wild_path],
        model_config=config["model"],
        amount_to_use=None,
        device=device,
        output_dir=args.output_dir
    )

    # Visualize metrics
    visualize_metrics(y_true, y_pred_label, y_pred_prob, output_dir=args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    IN_THE_WILD_DATASET_PATH = "./dataset/validate"
    parser.add_argument("--in_the_wild_path", type=str, default=IN_THE_WILD_DATASET_PATH)
    
    default_model_config = "configs/specrnet_balanced.yaml"
    parser.add_argument("--config", type=str, default=default_model_config)

    parser.add_argument("--cpu", "-c", help="Force using cpu", action="store_true")

    # Output directory for visualizations
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Directory to save visualizations")

    return parser.parse_args()


task.close()

if __name__ == "__main__":
    main(parse_args())
