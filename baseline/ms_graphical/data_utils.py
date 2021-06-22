import datasets
import copy
import utils

def load_and_encode_train_data_from_scrach(table_name, num_train, num_test, columns_list, operators_list, vals_list, card_list):
    table, _ = MakeTable(table_name)
    
    file_name_queries = "data/train"

    label = card_list


    # Get operator name dict
    operators = ['<=', '>=', '=']

    # # Get min and max values for each column
    # with open(file_name_column_min_max_vals, 'rU') as f:
    #     data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
    #     column_min_max_vals = {}
    #     for i, row in enumerate(data_raw):
    #         if i == 0:
    #             continue
    #         column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    column_min_max_vals = {}
    num_cols = len(table.columns)
    initial_range = []
    for col_idx in range(num_cols):
        column = table.columns[col_idx]
        all_col_vals = column.all_distinct_values

        if table_name in ['dmv', 'dmv-tiny'] and col_idx == 6:
            print(all_col_vals)
            col_min_val = all_col_vals[0] if all_col_vals[0] != 0 else all_col_vals[1]
            col_max_val = all_col_vals[-1]
        elif table_name == 'adult' and col_idx in [0, 3, 9, 10, 11]:
            col_min_val = all_col_vals[0]
            col_max_val = all_col_vals[-1]
        else:
            col_min_val = 0
            col_max_val = len(all_col_vals) - 1
        column_min_max_vals[column.name] = [float(col_min_val), float(col_max_val)]
        initial_range.append([float(0.), float(1.)])

    vals_list = discretize_vals(table_name, columns_list, vals_list, table)
    predicates = []
    l_len = len(columns_list)
    for i in range(l_len):
        temp = copy.deepcopy(initial_range)
        cols = columns_list[i]
        ops = operators_list[i]
        vals = vals_list[i]
        q_len = len(cols)
        for j in range(q_len):
            col_name = cols[j]
            col_id = table.ColumnIndex(col_name)
            op = ops[j]
            val = vals[j]
            norm_val = utils.normalize_data(val, col_name, column_min_max_vals)
            if op == '=':
                temp[col_id] = [norm_val, norm_val]
            elif op in ['<', '<=']:
                temp[col_id][1] = norm_val
            else:
                temp[col_id][0] = norm_val

        q_range = []
        for c_range in temp:
            q_range.append(c_range[0])
            q_range.append(c_range[1])
        predicates.append(q_range)

    # Get feature encoding and proper normalization

    label_norm, min_val, max_val, is_add_eps = utils.normalize_labels(label)


    #samples_train = samples_enc[:num_train]
    predicates_train = predicates[:num_train]
    labels_train = label_norm[:num_train]

    #samples_test = samples_enc[num_train:num_train + num_test]
    predicates_test = predicates[num_train:]
    labels_test = label_norm[num_train:]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))


    train_data = predicates_train
    test_data = predicates_test
    return column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_data, is_add_eps

def discretize_vals(table_name, columns_list, vals_list, table):
    res_vals_list = []
    list_len = len(columns_list)
    for i in range(list_len):
        temp = []
        cols = columns_list[i]
        vals = vals_list[i]
        q_len = len(cols)
        for j in range(q_len):
            col_name = cols[j]
            val = vals[j]
            col_idx = table.ColumnIndex(col_name)
            if table_name in ['dmv', 'dmv-tiny'] and col_idx == 6:
                if val != np.datetime64('NaT'):
                    temp.append(float(np.datetime64(val)))
                else:
                    temp.append(0)
            elif table_name == 'adult' and col_idx in [0,3,9,10,11]:
                temp.append(float(val))
            else:
                column = table.columns[col_idx]
                if isinstance(type(val), type(column.all_distinct_values[-1])) is False:
                    val = type(column.all_distinct_values[-1])(val)
                if isinstance(type(column.all_distinct_values[0]),
                              type(column.all_distinct_values[-1])) is False:
                    column.all_distinct_values[0] = type(
                        column.all_distinct_values[-1])(
                        column.all_distinct_values[0])

                temp.append(float(list(column.all_distinct_values).index(val)))
        res_vals_list.append(temp)
    return res_vals_list


def MakeTable(dataset):

    if dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif dataset == 'dmv':
        table = datasets.LoadDmv()
    elif dataset == 'covtype':
        table = datasets.LoadCovtype()
    elif dataset == 'kddcup':
        table = datasets.LoadKddcup()
    elif dataset == 'poker':
        table = datasets.LoadPoker()
    elif dataset == 'census':
        table = datasets.LoadCensus()

    return table, None

