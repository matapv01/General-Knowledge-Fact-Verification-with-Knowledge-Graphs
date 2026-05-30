import numpy as np

from .reasoning import TABLE_ORDER, TYPE_NAMES


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


def print_results_from_tags(model_name, is_correct_list, tag_lists):
    """Print results by counting every tag in each sample's types list.

    Note: BERT, GEAR, and evaluate_pgr.py use get_type_id() + print_results_from_flags
    instead (one type per claim, same priority as FactKG GEAR baseline).

    is_correct_list : list of booleans
    tag_lists       : list of lists, each inner list is the 'types' field of one sample
    """
    correct_by_name = {name: 0 for name in TYPE_NAMES.values()}
    total_by_name   = {name: 0 for name in TYPE_NAMES.values()}

    for is_correct, tags in zip(is_correct_list, tag_lists):
        for tag in tags:
            if tag in TYPE_NAMES:
                name = TYPE_NAMES[tag]
                total_by_name[name] += 1
                if is_correct:
                    correct_by_name[name] += 1

    print()
    print(f"Results on FactKG test set  [{model_name}]")
    print()
    print(f"{'Reasoning Type':<16}  {'Accuracy':>10}  {'Correct / Total':>15}")
    print("-" * 46)

    for _, type_name in TABLE_ORDER:
        correct = correct_by_name[type_name]
        total   = total_by_name[type_name]
        if total > 0:
            print(f"{type_name:<16}  {correct/total:>9.2%}  {correct:>6} / {total:<6}")
        else:
            print(f"{type_name:<16}  {'—':>10}  {'—':>15}")

    print("-" * 46)
    total_correct = sum(is_correct_list)
    total         = len(is_correct_list)
    overall_acc   = total_correct / total if total > 0 else 0
    print(f"{'Overall':<16}  {overall_acc:>9.2%}  {total_correct:>6} / {total:<6}")
    print()
