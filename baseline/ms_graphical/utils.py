import numpy as np
import torch

# Helper functions for data processing

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot

def get_column_encoding(table):
    columns = table.columns
    num_elements = len(table.columns)
    idx2column = [col.name for col in columns]
    column2idx = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(idx2column)}

    return column2idx, idx2column

def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
    min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                val = float(predicate[2])
                if val < min_max_vals[column_name][0]:
                    min_max_vals[column_name][0] = val
                if val > min_max_vals[column_name][1]:
                    min_max_vals[column_name][1] = val
    return min_max_vals


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return float(val_norm)


def normalize_labels(labels, min_val=None, max_val=None):
    is_add_eps = False
    for l in labels:
        if float(l) == 0.:
            is_add_eps = True
            break
    if is_add_eps:
        labels = np.array([np.log(float(l+1e-12)) for l in labels])
    else:
        labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val, is_add_eps


def unnormalize_labels(labels_norm, min_val, max_val, is_add_eps):
    labels_norm[labels_norm > 1] = 1
    labels_norm[labels_norm < 0] = 0

    labels = (labels_norm * (max_val - min_val)) + min_val
    if is_add_eps:
        return np.maximum(np.exp(labels) - 1e-12, 0.)
    else:
        return np.exp(labels)

def unnormalize_labels_torch(labels_norm, min_val, max_val, is_add_eps):
    labels_norm[labels_norm > 1] = 1
    labels_norm[labels_norm < 0] = 0

    labels = (labels_norm * (max_val - min_val)) + min_val
    if is_add_eps:
        return torch.max(torch.exp(labels) - 1e-12, 0.0)
    else:
        return torch.exp(labels)

def encode_samples(samples, tables):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc


def encode_data(predicates, column_min_max_vals, column2vec, op2vec):
    predicates_enc = []
    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)
                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))
            predicates_enc[i].append(pred_vec)
    return predicates_enc

def get_qerror(pred, label):
    qerror = np.maximum(pred/label, label/pred)
    return qerror