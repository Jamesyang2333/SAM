import math
import bisect
import itertools
import time
from scipy.io import savemat
from scipy.optimize import minimize
import numpy as np
import networkx as nx
from data_loader import load_dataset_join
import datasets_join


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


def get_var_idx(cols, ops, vals, interval_list, clique, offset, loaded_tables):
    var_idx_list = [[] for col in clique]

    # print(ops)
    # print(cols)
    # print(vals)
    for i in range(len(ops)):
        col = cols[i]
        col_idx = clique.index(col)
        val = vals[i]
        var_idx_list[col_idx] = list(range(len(interval_list[col])))
        if ops[i] == "=":
            var_idx_list[col_idx] = [interval_list[col].index(val)]
        elif ops[i] == ">=":
            var_idx_list[col_idx] = var_idx_list[col_idx][var_idx_list[col_idx].index(interval_list[col].index(val)):]
        elif ops[i] == ">":
            table_name, col_name = col.split('.')
            table = loaded_tables[table_name]
            column = table.columns[table.ColumnIndex(col_name)]
            all_distinct_values = list(column.all_distinct_values)
            if all_distinct_values.index(val) < len(all_distinct_values) - 1:
                var_idx_list[col_idx] = var_idx_list[col_idx][var_idx_list[col_idx].index(interval_list[col].index(all_distinct_values[all_distinct_values.index(val) + 1])):]
        elif ops[i] == "<=":
            table_name, col_name = col.split('.')
            table = loaded_tables[table_name]
            column = table.columns[table.ColumnIndex(col_name)]
            all_distinct_values = list(column.all_distinct_values)
            if all_distinct_values.index(val) < len(all_distinct_values) - 1:
                var_idx_list[col_idx] = var_idx_list[col_idx][:var_idx_list[col_idx].index(interval_list[col].index(all_distinct_values[all_distinct_values.index(val) + 1]))]
        elif ops[i] == "<":
            var_idx_list[col_idx] = var_idx_list[col_idx][:var_idx_list[col_idx].index(interval_list[col].index(val))]

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


join_tables = [
        'cast_info', 'movie_companies', 'movie_info', 'movie_keyword', 'title',
        'movie_info_idx'
    ]

join_col = ['id', 'movie_id']

dataset = 'imdb'
# New datasets should be loaded here.
assert dataset in ['imdb']
if dataset == 'imdb':
    print('Training on Join({})'.format(join_tables))

    loaded_tables = {}
    for t in join_tables:
        print('Loading', t)
        table = datasets_join.LoadImdb(t, use_cols='content')
        
        table.data.info()
        # loaded_tables[name_dict[t]] = table
        loaded_tables[t] = table


train_data_raw = load_dataset_join("../queries/mscn_queries_neurocard_format.csv", join_tables, loaded_tables)
print("Total number of queries: {}".format(len(train_data_raw)))

idx_list_filter = []
join_name_set = ['movie_info_idx,title', 'title', 'movie_keyword,title', 'cast_info,title', 'movie_companies,title', 'movie_info,title']

# classify queries in terms of join name
query_by_join = {}
for name in join_name_set:
    query_by_join[name] = []

# get rid of queries with filter on both column 0 and 14
for i in range(len(train_data_raw)):
    if len(train_data_raw[i]["tables"]) == 1 and train_data_raw[i]["tables"][0] == "title":
        idx_list_filter.append(i)

    if len(train_data_raw[i]["tables"]) == 2:
        idx_list_filter.append(i)

train_data_raw = [train_data_raw[i] for i in idx_list_filter]
print("Total number of queries after filter: {}".format(len(train_data_raw)))


# train_num = len(train_data_raw["card"])
train_num = 20

for i in range(train_num):
    if len(train_data_raw[i]["tables"]) == 1:
        idx_list_filter.append(i)
        query_by_join['title'].append(train_data_raw[i])

    if len(train_data_raw[i]["tables"]) == 2:
        idx_list_filter.append(i)
        train_data_raw[i]["tables"].sort()
        query_by_join[','.join(train_data_raw[i]["tables"])].append(train_data_raw[i])


graph_set = {}
interval_dict_set = {}
interval_list_set = {}
interval_dict_idx_set = {}
interval_list_idx_set = {}

for name in join_name_set:
    graph_set[name] = nx.Graph()
    interval_dict_set[name] = {}
    interval_list_set[name] = {}
    interval_dict_idx_set[name] = {}
    interval_list_idx_set[name] = {}

for name in join_name_set:
    for table_name in name.split(','):
        table = loaded_tables[table_name]
        for col in table.columns:
            if not col in join_col:
                col_name = table_name + "." + col.name
                graph_set[name].add_node(col_name)
                interval_dict_set[name][col_name] = set()
                interval_dict_idx_set[name][col_name] = set()

for i in range(train_num):

    tables = train_data_raw[i]["tables"]
    join_name = ','.join(train_data_raw[i]["tables"]) 

    cols = train_data_raw[i]["predicates"][0]
    ops = train_data_raw[i]["predicates"][1]
    vals = train_data_raw[i]["predicates"][2]
    n_cols = len(cols)

    # construct the column dependency graphs
    for j in range(n_cols):
        for k in range(j+1, n_cols):
            graph_set[join_name].add_edge(cols[j], cols[k])
    
    for j in range(n_cols):
        table_name, col = cols[j].split('.')
        table = loaded_tables[table_name]
        col_idx = table.ColumnIndex(col)
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
                interval_dict_set[join_name][cols[j]].add(all_distinct_values[idx])
                interval_dict_idx_set[join_name][cols[j]].add(idx)
            if idx + 1 < len(all_distinct_values):
                if all_distinct_values[idx + 1] != float("nan"):
                    interval_dict_set[join_name][cols[j]].add(all_distinct_values[idx + 1])
                    interval_dict_idx_set[join_name][cols[j]].add(idx + 1)
        if ops[j] == ">=":
            if all_distinct_values[idx] != float("nan"):
                interval_dict_set[join_name][cols[j]].add(all_distinct_values[idx])
                interval_dict_idx_set[join_name][cols[j]].add(idx)
        if ops[j] == ">":
            if idx + 1 < len(all_distinct_values):
                if all_distinct_values[idx] != float("nan"):
                    interval_dict_set[join_name][cols[j]].add(all_distinct_values[idx + 1])
                    interval_dict_idx_set[join_name][cols[j]].add(idx + 1)
        if ops[j] == "<=":
            if idx + 1 < len(all_distinct_values):
                if all_distinct_values[idx + 1] != float("nan"):
                    interval_dict_set[join_name][cols[j]].add(all_distinct_values[idx + 1])
                    interval_dict_idx_set[join_name][cols[j]].add(idx + 1)
        if ops[j] == "<":
            if all_distinct_values[idx] != float("nan"):
                interval_dict_set[join_name][cols[j]].add(all_distinct_values[idx])
                interval_dict_idx_set[join_name][cols[j]].add(idx)

for name in join_name_set:
    for table_name in name.split(','):
        table = loaded_tables[table_name]
        for col in table.Columns():
            all_distinct_values = list(col.all_distinct_values)
            col_name = table_name + "." + col.name
            if all_distinct_values[0] != all_distinct_values[0]:
                interval_dict_set[name][col_name].add(all_distinct_values[1])
                interval_dict_idx_set[name][col_name].add(1)
            else:
                interval_dict_set[name][col_name].add(all_distinct_values[0])
                interval_dict_idx_set[name][col_name].add(0)
# create interval list for each column in sorted order

# every column has len(interval_list[col.name]) - 1 intervals and an extra for nan

norm = 2528312
for name in join_name_set:
    for table_name in name.split(','):
        table = loaded_tables[table_name]
        for col in table.Columns():
            col_name = table_name + "." + col.name
            interval_list_set[name][col_name] = []
            interval_list_idx_set[name][col_name] = []
            for val in interval_dict_set[name][col_name]:
                interval_list_set[name][col_name].append(val)
            for val in interval_dict_idx_set[name][col_name]:
                interval_list_idx_set[name][col_name].append(val)
            interval_list_set[name][col_name].sort()
            interval_list_idx_set[name][col_name].sort()


sample_view_dict = {}
for name in join_name_set:
    sample_view_dict[name] = {}
    
for name in join_name_set:
    print("join tables: {}".format(name))
    print("number of nodes: {}".format(graph_set[name].number_of_nodes()))
    print("number of edges: {}".format(graph_set[name].number_of_edges()))
    print("graph is cordal: {}".format(nx.algorithms.chordal.is_chordal(graph_set[name])))
    clique_set = nx.algorithms.chordal.chordal_graph_cliques(graph_set[name])
    print(clique_set)
    clique_list = []
    for clique in clique_set:
        current_list = []
        for col in clique:
            current_list.append(col)
        current_list.sort()
        clique_list.append(current_list)

    total_nvar = 0
    nvar_list = []
    for clique in clique_list:
        count = 1
        for col in clique:
            count *= (len(interval_list_set[name][col]) + 1)
        total_nvar += count
        nvar_list.append(count)
        print("varibles required for clique: {}".format(count))

    nvar_cum_sum_list = [0 for i in range(len(clique_list))]
    for i in range(len(clique_list) - 1):
        nvar_cum_sum_list[i + 1] = nvar_cum_sum_list[i] + nvar_list[i]

    # print(nvar_cum_sum_list)
    # print(clique_list)

    if name == "title":
        # marginal probabilities sum to 1
        a_marginal = np.zeros((len(clique_list), total_nvar))
        b_marginal = np.ones(len(clique_list))
        cum_sum = 0
        for i in range(len(clique_list)):
            a_marginal[i][cum_sum:cum_sum+nvar_list[i]]=1
            b_marginal[i] = loaded_tables['title'].cardinality / norm
            cum_sum += nvar_list[i]
    else:
        # total cardinality for join not known, but the sum of marginal probabilities for each clique should sum to 1
        a_marginal = np.zeros((len(clique_list) - 1, total_nvar))
        b_marginal = np.ones(len(clique_list) - 1)
        for i in range(len(clique_list) - 1):
            a_marginal[i][nvar_cum_sum_list[i]:nvar_cum_sum_list[i] + nvar_list[i]] = 1
            a_marginal[i][nvar_cum_sum_list[i + 1]:nvar_cum_sum_list[i + 1] + nvar_list[i + 1]] = -1
            b_marginal[i] = 0
    
    # print(a_marginal)
    # print(b_marginal)

    # query conditions should satisfy
    a_query = np.zeros((len(query_by_join[name]), total_nvar))
    b_query = np.zeros(len(query_by_join[name]))
    # print(query_by_join[name])
    for i in range(len(query_by_join[name])):
        b_query[i] = query_by_join[name][i]['label'] / norm
        clique_idx = get_clique_idx(query_by_join[name][i]['predicates'][0], clique_list)
        offset = nvar_cum_sum_list[clique_idx]
        all_idx_list = get_var_idx(query_by_join[name][i]['predicates'][0], query_by_join[name][i]['predicates'][1], query_by_join[name][i]['predicates'][2], interval_list_set[name], clique_list[clique_idx], offset, loaded_tables)
        # print(train_data_raw["column"][i])
        # print(train_data_raw["operator"][i])
        # print(train_data_raw["val"][i])
        # for col in train_data_raw["column"][i]:
        #     print(interval_list[col])
        # print(all_idx_list)
        a_query[i][all_idx_list] = 1

    # print(interval_list_set[name])
    # print(a_query)
    # print(b_query)

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
                    col_vars_list.append(interval_list_set[name][col])
                all_var_comb_list = itertools.product(*col_vars_list)
                for comb in all_var_comb_list:
                    all_idx_list_1 = get_var_idx(cols, ["=" for col in cols], comb, interval_list_set[name], clique_list[i], nvar_cum_sum_list[i], loaded_tables)
                    all_idx_list_2 = get_var_idx(cols, ["=" for col in cols], comb, interval_list_set[name], clique_list[j], nvar_cum_sum_list[j], loaded_tables)
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
    # print(x)

    for clique in clique_list:
        for col in clique:
            sample_view_dict[name][col] = []

    n_samples = 1

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
                val_idx, val = get_val_from_idx(interval_list_set[name], clique_list[i], sample_idx)
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
                idx_list = get_var_idx(common_cols, ["=" for _ in common_cols], col_vars_list, interval_list_set[name], clique_list[i], nvar_cum_sum_list[i], loaded_tables)
                prob = np.copy(x[idx_list])
                prob /= prob.sum()
                sample_idx = np.random.choice(len(prob), p=prob)
                sample_idx = idx_list[sample_idx] - nvar_cum_sum_list[i]
                val_idx, val = get_val_from_idx(interval_list_set[name], clique_list[i], sample_idx)
                for col in val:
                    val_dict[col] = val[col]
                # val_list.append(val)
                # print(val_idx)
                # print(val)
        
        for col in val_dict:
                    sample_view_dict[name][col].append(val_dict[col])

        end = time.time()
        print("Total time takes for sampling {} tuples: {}".format(n_samples, end - start))