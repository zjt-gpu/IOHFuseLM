import numpy as np
import os

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)

    return mae, mse

def low_MAE(pred, true, label):
    if np.sum(label) == 0:  
        return 0
    else:
        return np.sum(np.abs((pred - true) * label)) / np.sum(label) 

def low_MSE(pred, true, label):
    if np.sum(label) == 0:  
        return 0
    else:
        return np.sum(np.abs((pred - true) * label) ** 2) / np.sum(label)  


def low_metric(pred, true, label):
    low_mae = low_MAE(pred, true, label)
    low_mse = low_MSE(pred, true, label)

    return low_mae, low_mse



class EvaluationRecorder:
    def __init__(self, args):
        self.args = args

    def evaluate_and_save(self, fine_tune_fn):
        (
            inverse_mses, inverse_maes,
            inverse_low_mses, inverse_low_maes,
            accuracys, specificitys, precisions, recalls, f1s, aucs,
            num_pos_labels, num_neg_labels, num_pos_preds, num_neg_preds
        ) = fine_tune_fn(self.args)

        inverse_mses = np.array(inverse_mses)
        inverse_maes = np.array(inverse_maes)
        inverse_low_mses = np.array(inverse_low_mses)
        inverse_low_maes = np.array(inverse_low_maes)
        accuracys = np.array(accuracys)
        specificitys = np.array(specificitys)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        aucs = np.array(aucs)

        metrics = {
            "inverse_low_mse_mean": np.mean(inverse_low_mses), "inverse_low_mse_std": np.std(inverse_low_mses),
            "inverse_low_mae_mean": np.mean(inverse_low_maes), "inverse_low_mae_std": np.std(inverse_low_maes),
            "accuracy": np.mean(accuracys),
            "specificity": np.mean(specificitys),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "F1": np.mean(f1s),
            "AUC": np.mean(aucs),
            "num_pos_labels": num_pos_labels,
            "num_neg_labels": num_neg_labels,
            "num_pos_preds": num_pos_preds,
            "num_neg_preds": num_neg_preds
        }

        #self._print_metrics(metrics)

        self._save_metrics(metrics)

    def _print_metrics(self, metrics):
        print(f"low_mse_mean = {metrics['low_mse_mean']:.4f}, low_mse_std = {metrics['low_mse_std']:.4f}")
        print(f"low_mae_mean = {metrics['low_mae_mean']:.4f}, low_mae_std = {metrics['low_mae_std']:.4f}")
        print(f"inverse_low_mse_mean = {metrics['inverse_low_mse_mean']:.4f}, inverse_low_mse_std = {metrics['inverse_low_mse_std']:.4f}")
        print(f"inverse_low_mae_mean = {metrics['inverse_low_mae_mean']:.4f}, inverse_low_mae_std = {metrics['inverse_low_mae_std']:.4f}")
        print(f"accuracy = {metrics['accuracy']:.4f}, specificity = {metrics['specificity']:.4f}, "
              f"precision = {metrics['precision']:.4f}, recall = {metrics['recall']:.4f}, "
              f"F1 = {metrics['F1']:.4f}, AUC = {metrics['AUC']:.4f}")

    def _save_metrics(self, metrics):
        save_dir = f"result/{self.args.model}/{self.args.dataset_name}/seq_len_{self.args.seq_len}_pred_len_{self.args.pred_len}_sample_{self.args.sampling_rate}"
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, "all_result.txt")

        with open(output_path, "w") as f:
            for key, value in metrics.items():
                if "mean" in key or "std" in key or key in ["accuracy", "specificity", "precision", "recall", "F1", "AUC"]:
                    f.write(f"{key} = {value:.4f}\n")
                else:
                    f.write(f"{key} = {int(value)}\n")

        print(f"Results saved to {output_path}")