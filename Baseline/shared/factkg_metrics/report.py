import numpy as np

from .reasoning import TABLE_ORDER


def print_results(model_name, labels, preds, type_ids):
    """Print per-reasoning-type accuracy matching Table 3 in the FactKG paper.

    labels   : ground-truth labels (0 or 1) as list or numpy array
    preds    : predicted labels (0 or 1) as list or numpy array
    type_ids : integer type IDs (0-4) as list or numpy array
    """
    labels   = np.array(labels)
    preds    = np.array(preds)
    type_ids = np.array(type_ids)

    print()
    print(f"Results on FactKG test set  [{model_name}]")
    print()
    print(f"{'Reasoning Type':<16}  {'Accuracy':>10}  {'Correct / Total':>15}")
    print("-" * 46)

    for type_id, type_name in TABLE_ORDER:
        mask = type_ids == type_id
        if mask.any():
            correct = int((labels[mask] == preds[mask]).sum())
            total   = int(mask.sum())
            print(f"{type_name:<16}  {correct/total:>9.2%}  {correct:>6} / {total:<6}")
        else:
            print(f"{type_name:<16}  {'—':>10}  {'—':>15}")

    print("-" * 46)
    total_correct = int((labels == preds).sum())
    total         = len(labels)
    print(f"{'Overall':<16}  {total_correct/total:>9.2%}  {total_correct:>6} / {total:<6}")
    print()


def print_results_from_flags(model_name, is_correct, type_ids):
    """Print results using a pre-computed correctness array.

    Used when the training loop already computes (pred == gt) and stores it,
    rather than storing raw predictions separately (e.g. GEAR baseline).

    is_correct : boolean list/array, True if the sample was predicted correctly
    type_ids   : integer type IDs (0-4) as list or numpy array
    """
    is_correct = np.array(is_correct, dtype=bool)
    type_ids   = np.array(type_ids)

    print()
    print(f"Results on FactKG test set  [{model_name}]")
    print()
    print(f"{'Reasoning Type':<16}  {'Accuracy':>10}  {'Correct / Total':>15}")
    print("-" * 46)

    for type_id, type_name in TABLE_ORDER:
        mask = type_ids == type_id
        if mask.any():
            correct = int(is_correct[mask].sum())
            total   = int(mask.sum())
            print(f"{type_name:<16}  {correct/total:>9.2%}  {correct:>6} / {total:<6}")
        else:
            print(f"{type_name:<16}  {'—':>10}  {'—':>15}")

    print("-" * 46)
    total_correct = int(is_correct.sum())
    total         = len(is_correct)
    print(f"{'Overall':<16}  {total_correct/total:>9.2%}  {total_correct:>6} / {total:<6}")
    print()
