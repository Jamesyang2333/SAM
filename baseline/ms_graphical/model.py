import math
import bisect
import itertools
import time

import numpy as np
import pandas as pd
import networkx as nx
from numpy.core.fromnumeric import take
from scipy.io import savemat
from scipy.optimize import minimize
import argparse

import datasets
from data_loader import load_dataset
from evaluate import Query

def get_qerror(pred, label):
    qerror = np.mean(np.maximum(pred/label, label/pred))
    return qerror

def get_clique_idx(cols, clique_list):
    for i in range(len(clique_list)):
        containing_clique = True
        for col in cols:
            if not col in clique_list[i]:
                containing_clique = False
                break
        if containing_clique:
            return i

def get_clique_prob(cols, vals, interval_list, interval_list_idx, clique, offset, x):
    var_idx_list = []
    # all values for possible for column not filtered, including null
    for col in clique:
        var_idx_list.append(list(range(len(interval_list[col]) + 1)))
    scale_factor = 1
    for i in range(len(cols)):
        col = cols[i]
        col_idx = clique.index(col)
        val = vals[i]
        if val != val:
            var_idx_list[i] = [len(interval_list[col])]
        else:
            pos = bisect.bisect(interval_list[col], val) 
            var_idx_list[col_idx] = [pos - 1]
            if pos == len(interval_list_idx[col]):
                column = table.columns[table.ColumnIndex(col)]
                all_distinct_values = list(column.all_distinct_values)
                scale_factor *= (len(all_distinct_values) - interval_list_idx[col][pos - 1])
            else:
                scale_factor *= (interval_list_idx[col][pos] - interval_list_idx[col][pos - 1])

    cum_product_list = [1 for i in range(len(clique))]
    cum_product_list[-1] = 1
    for i in range(1, len(clique)):
        cum_product_list[len(clique) - i - 1] = cum_product_list[len(clique) - i] * (len(interval_list[clique[len(clique) - i]]) + 1)
    all_comb_list = itertools.product(*var_idx_list)

    def get_idx(comb):
        sum = 0
        for i in range(len(clique)):
            sum += comb[i] * cum_product_list[i]
        return sum

    all_idx_list = [get_idx(comb) + offset for comb in all_comb_list]
    prob = np.sum(x[all_idx_list])

    # scale down the probability
    prob = prob / scale_factor

    return prob


def get_var_idx(cols, ops, vals, interval_list, clique, offset):
    var_idx_list = [[] for col in clique]

    # print(ops)
    # print(cols)
    # print(vals)
    # print(var_idx_list[12])
    for i in range(len(ops)):
        col = cols[i]
        col_idx = clique.index(col)
        val = vals[i]
        var_idx_list[col_idx] = list(range(len(interval_list[col])))
        if ops[i] == "=":
            # when val is nan, assign the largest idx
            if val != val:
                var_idx_list[col_idx] = [len(interval_list[col])]
            else:
                var_idx_list[col_idx] = [interval_list[col].index(val)]
        elif ops[i] == ">=":
            var_idx_list[col_idx] = var_idx_list[col_idx][var_idx_list[col_idx].index(interval_list[col].index(val)):]
        else:
            column = table.columns[table.ColumnIndex(col)]
            all_distinct_values = list(column.all_distinct_values)
            if all_distinct_values.index(val) < len(all_distinct_values) - 1:
                var_idx_list[col_idx] = var_idx_list[col_idx][:var_idx_list[col_idx].index(interval_list[col].index(all_distinct_values[all_distinct_values.index(val) + 1]))]

    # for columns not filtered, all variables include nan are possible
    for i in range(len(clique)):
        if not var_idx_list[i]:
            var_idx_list[i] = list(range(len(interval_list[clique[i]]) + 1))

        
    cum_product_list = [1 for i in range(len(clique))]
    cum_product_list[-1] = 1
    for i in range(1, len(clique)):
        cum_product_list[len(clique) - i - 1] = cum_product_list[len(clique) - i] * (len(interval_list[clique[len(clique) - i]]) + 1)
    all_comb_list = itertools.product(*var_idx_list)

    def get_idx(comb):
        sum = 0
        for i in range(len(clique)):
            sum += comb[i] * cum_product_list[i]
        return sum

    all_idx_list = [get_idx(comb) + offset for comb in all_comb_list]
    return all_idx_list


def get_val_from_idx(interval_list, clique, idx):
    cum_product_list = [1 for i in range(len(clique))]
    cum_product_list[-1] = 1
    for i in range(1, len(clique)):
        cum_product_list[len(clique) - i - 1] = cum_product_list[len(clique) - i] * (len(interval_list[clique[len(clique) - i]]) + 1)
    result = {}
    result_idx = {}
    # print(idx)
    # for col in clique:
    #     print(interval_list[col])
    for i in range(len(clique)):
        col = clique[i]
        current_idx = idx // cum_product_list[i]
        result_idx[col] = current_idx
        # print(i)
        # print(current_idx)
        # print(cum_product_list)
        if current_idx == len(interval_list[col]):
            result[clique[i]] = float('nan')
        else:
            result[clique[i]] = interval_list[col][current_idx]
        idx = idx % cum_product_list[i]

    return result_idx, result

parser = argparse.ArgumentParser()
parser.add_argument("--train_num", help="number of queries used for generation", type=int, default=50)

args = parser.parse_args()


train_data_raw = load_dataset("census", "/home_nfs/jingyi/db_generation/queries/adult_10000_2cols.txt")
idx_list_filter = []
# get rid of queries with filter on both column 0 and 14
for i in range(len(train_data_raw["card"])):
    if not ((0 in train_data_raw["column"][i]) and (14 in train_data_raw["column"][i])):
        idx_list_filter.append(i)
train_data_raw["card"] = [train_data_raw["card"][i] for i in idx_list_filter]
train_data_raw["column"] = [train_data_raw["column"][i] for i in idx_list_filter]
train_data_raw["operator"] = [train_data_raw["operator"][i] for i in idx_list_filter]
train_data_raw["val"] = [train_data_raw["val"][i] for i in idx_list_filter]

# train_num = len(train_data_raw["card"])
train_num = args.train_num


print(train_num)

G = nx.Graph()
table = datasets.LoadCensus()
interval_dict = {}
interval_list = {}
interval_dict_idx = {}
interval_list_idx = {}

for col in table.columns:
    G.add_node(col.name)
    interval_dict[col.name] = set()
    interval_dict_idx[col.name] = set()

for i in range(train_num):

    cols = train_data_raw["column"][i]
    # print(i)
    ops = train_data_raw["operator"][i]
    vals = train_data_raw["val"][i]
    n_cols = len(cols)

    # construct the column dependency graphs
    for j in range(n_cols):
        for k in range(j+1, n_cols):
            G.add_edge(cols[j], cols[k])
    
    for j in range(n_cols):
        col_idx = table.ColumnIndex(cols[j])
        column = table.columns[col_idx]
        all_distinct_values = list(column.all_distinct_values)
        if isinstance(type(vals[j]), type(all_distinct_values[-1])) is False:
                vals[j] = type(all_distinct_values[-1])(vals[j])
        if vals[j] not in all_distinct_values:
            print(j)
            print(vals[j])
            print(type(vals[j]))
            print(all_distinct_values)
            print(type(all_distinct_values[5]))
        idx = all_distinct_values.index(vals[j])
        if ops[j] == "=":
            if all_distinct_values[idx] != float("nan"):
                interval_dict[cols[j]].add(all_distinct_values[idx])
                interval_dict_idx[cols[j]].add(idx)
            if idx + 1 < len(all_distinct_values):
                if all_distinct_values[idx + 1] != float("nan"):
                    interval_dict[cols[j]].add(all_distinct_values[idx + 1])
                    interval_dict_idx[cols[j]].add(idx + 1)
        if ops[j] == ">=":
            if all_distinct_values[idx] != float("nan"):
                interval_dict[cols[j]].add(all_distinct_values[idx])
                interval_dict_idx[cols[j]].add(idx)
        if ops[j] == "<=":
            if idx + 1 < len(all_distinct_values):
                if all_distinct_values[idx + 1] != float("nan"):
                    interval_dict[cols[j]].add(all_distinct_values[idx + 1])
                    interval_dict_idx[cols[j]].add(idx + 1)

for col in table.Columns():
    all_distinct_values = list(col.all_distinct_values)
    if all_distinct_values[0] != all_distinct_values[0]:
        interval_dict[col.name].add(all_distinct_values[1])
        interval_dict_idx[col.name].add(1)
    else:
        interval_dict[col.name].add(all_distinct_values[0])
        interval_dict_idx[col.name].add(0)
# create interval list for each column in sorted order

# every column has len(interval_list[col.name]) - 1 intervals and an extra for nan
for col in table.columns:
    interval_list[col.name] = []
    interval_list_idx[col.name] = []
    for val in interval_dict[col.name]:
        interval_list[col.name].append(val)
    for val in interval_dict_idx[col.name]:
        interval_list_idx[col.name].append(val)
    interval_list[col.name].sort()
    interval_list_idx[col.name].sort()

    

print("number of nodes: {}".format(G.number_of_nodes()))
print("number of edges: {}".format(G.number_of_edges()))
print(G.edges)
print("graph is cordal: {}".format(nx.algorithms.chordal.is_chordal(G)))
clique_set = nx.algorithms.chordal.chordal_graph_cliques(G)
print(clique_set)
clique_list = []
for clique in clique_set:
    current_list = []
    for col in clique:
        current_list.append(col)
    current_list.sort()
    clique_list.append(current_list)
clique_list.sort()

total_nvar = 0
nvar_list = []
for clique in clique_list:
    count = 1
    for col in clique:
        count *= (len(interval_list[col]) + 1)
    total_nvar += count
    nvar_list.append(count)
    print("varibles required for clique: {}".format(count))

nvar_cum_sum_list = [0 for i in range(len(clique_list))]
for i in range(len(clique_list) - 1):
    nvar_cum_sum_list[i + 1] = nvar_cum_sum_list[i] + nvar_list[i]

print(nvar_cum_sum_list)
# marginal probabilities sum to 1
a_marginal = np.zeros((len(clique_list), total_nvar))
b_marginal = np.ones(len(clique_list))
cum_sum = 0
for i in range(len(clique_list)):
    a_marginal[i][cum_sum:cum_sum+nvar_list[i]]=1
    cum_sum += nvar_list[i]


# query conditions should satisfy
a_query = np.zeros((train_num, total_nvar))
b_query = np.zeros(train_num)
for i in range(train_num):
    b_query[i] = train_data_raw["card"][i]
    clique_idx = get_clique_idx(train_data_raw["column"][i], clique_list)
    offset = nvar_cum_sum_list[clique_idx]
    all_idx_list = get_var_idx(train_data_raw["column"][i], train_data_raw["operator"][i], train_data_raw["val"][i], interval_list, clique_list[clique_idx], offset)
    # print(train_data_raw["column"][i])
    # print(train_data_raw["operator"][i])
    # print(train_data_raw["val"][i])
    # for col in train_data_raw["column"][i]:
    #     print(interval_list[col])
    # print(all_idx_list)
    a_query[i][all_idx_list] = 1

a = np.concatenate((a_marginal, a_query), axis=0)
b = np.concatenate((b_marginal, b_query), axis=0)

# marginal distribution can be computed in two ways and they should give the same result
for i in range(len(clique_list)):
    for j in range(i+1, len(clique_list)):
        common_cols = set(clique_list[i]) & set(clique_list[j])
        if common_cols:
            cols = list(common_cols)
            col_vars_list = []
            for col in cols:
                col_vars_list.append(interval_list[col])
            all_var_comb_list = itertools.product(*col_vars_list)
            for comb in all_var_comb_list:
                all_idx_list_1 = get_var_idx(cols, ["=" for col in cols], comb, interval_list, clique_list[i], nvar_cum_sum_list[i])
                all_idx_list_2 = get_var_idx(cols, ["=" for col in cols], comb, interval_list, clique_list[j], nvar_cum_sum_list[j])
                a_temp = np.zeros((1, total_nvar))
                b_temp = np.zeros(1)   
                a_temp[0][all_idx_list_1] = 1
                a_temp[0][all_idx_list_2] = -1
                a = np.concatenate((a, a_temp), axis=0)
                b = np.concatenate((b, b_temp), axis=0)

mdic = {"a": a, "b": b}
savemat("matlab_matrix.mat", mdic)

print(a.shape)
print(b.shape)
# print(b)
# p, res, rnk, s = lstsq(a, b)
# print(p)
# print(res)
# print(rnk)
# print(np.linalg.norm(np.matmul(a, p) - b))


start = time.time()

fun = lambda x: np.linalg.norm(np.dot(a,x)-b)
# xo = np.linalg.solve(a,b)
sol = minimize(fun, np.zeros(a.shape[1]), method='SLSQP', bounds=[(1e-9,None) for x in range(a.shape[1])])

end = time.time()
print("Total time takes for solving LP: {}".format(end - start))

print("object value after minimization: {}".format(sol['fun']))
print(sol['message'])


x = sol['x']

cross_entropy = 0
nan_count = 0
for pos in range(table.cardinality):

    contain_nan = False
    for col in table.columns:
        if table.data.loc[pos][col.name] != table.data.loc[pos][col.name]:
            contain_nan = True
    
    if contain_nan:
        nan_count += 1

    total_prob = 1
    for i in range(len(clique_list)):
        vals = []
        for col in clique_list[i]:
            vals.append(table.data.loc[pos][col])
        cur_prob = get_clique_prob(clique_list[i], vals, interval_list, interval_list_idx, clique_list[i], nvar_cum_sum_list[i], x)
        total_prob *= cur_prob

    for i in range(len(clique_list)):
        for j in range(i+1, len(clique_list)):
            common_cols = list(set(clique_list[i]) & set(clique_list[j]))
            common_cols.sort()
            if common_cols:
                vals = []
                for col in common_cols:
                    vals.append(table.data.loc[pos][col])
                cur_prob = get_clique_prob(common_cols, vals, interval_list, interval_list_idx, clique_list[i], nvar_cum_sum_list[i], x)
                total_prob /= cur_prob
    
    cross_entropy += (1/table.cardinality*math.log(total_prob))
print("cross entropy: {}".format(-cross_entropy))
print("total numebr of rows: {}".format(table.cardinality))
print("number of rows containing nan: {}".format(nan_count))


print(clique_list)
sample_dict = {}
for col in table.columns:
    sample_dict[col.name] = []

n_samples = table.cardinality

start = time.time()

for _ in range(n_samples):
    val_dict = {}
    # val_list = []
    for i in range(len(clique_list)):
        if i == 0 or not (set(clique_list[i]) & set(clique_list[i-1])):
            # if it's the last clique, get all value from nvar_cum_sum_list[i] onward
            if i == len(clique_list) - 1:
                prob = np.copy(x[nvar_cum_sum_list[i]:])
            else:
                prob = np.copy(x[nvar_cum_sum_list[i]: nvar_cum_sum_list[i+1]])
            prob /= prob.sum()
            # print(sum(prob))
            sample_idx = np.random.choice(nvar_list[i], p=prob)
            val_idx, val = get_val_from_idx(interval_list, clique_list[i], sample_idx)
            for col in val:
                val_dict[col] = val[col]
            # val_list.append(val)
            # print(val_idx)
            # print(val)

        else:
            common_cols = list(set(clique_list[i]) & set(clique_list[i-1]))
            col_vars_list = []
            for col in common_cols:
                col_vars_list.append(val_dict[col])
            idx_list = get_var_idx(common_cols, ["=" for _ in common_cols], col_vars_list, interval_list, clique_list[i], nvar_cum_sum_list[i])
            prob = np.copy(x[idx_list])
            prob /= prob.sum()
            sample_idx = np.random.choice(len(prob), p=prob)
            sample_idx = idx_list[sample_idx] - nvar_cum_sum_list[i]
            val_idx, val = get_val_from_idx(interval_list, clique_list[i], sample_idx)
            for col in val:
                val_dict[col] = val[col]
            # val_list.append(val)
            # print(val_idx)
            # print(val)
    
    for col in val_dict:
                sample_dict[col].append(val_dict[col])

sample_frame = pd.DataFrame.from_dict(sample_dict)
sample_table = datasets.LoadCensus(sample_frame)

end = time.time()
print("Total time takes for sampling {} tuples: {}".format(n_samples, end - start))

card_pred = []
for i in range(train_num):
    cols = [sample_table.columns[sample_table.ColumnIndex(col)] for col in train_data_raw['column'][i]]
    ops = train_data_raw['operator'][i]
    vals = train_data_raw['val'][i]
    est = Query(sample_table, cols, ops, vals)
    card_pred.append(est / sample_table.cardinality)

print("q error on generated dataset:  {}".format(get_qerror(np.array(card_pred), np.array(train_data_raw['card'][:train_num]))))

