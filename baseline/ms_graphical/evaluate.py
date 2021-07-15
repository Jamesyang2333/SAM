import numpy as np
import pandas as pd
import random
import time

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


def Query(table, columns, operators, vals, return_masks=False, return_crad_and_masks=False):
    assert len(columns) == len(operators) == len(vals)
    bools = None
    for c, o, v in zip(columns, operators, vals):
        if table.name in ['Adult', 'Census']:
            inds = [False] * table.cardinality
            inds = np.array(inds)
            is_nan = pd.isnull(c.data)
            if np.any(is_nan):
                inds[~is_nan] = OPS[o](c.data[~is_nan], v)
            else:
                inds = OPS[o](c.data, v)
        else:
            inds = OPS[o](c.data, v)

        if bools is None:
            bools = inds
        else:
            bools &= inds
    c = bools.sum()
    if return_masks:
        return bools
    elif return_crad_and_masks:
        return c, bools
    return c


def ComputeCE(gt_table, gen_table, gt_caches, eps=1e-9):

    # fill nan value with -1 for both data frames
    gt_table = gt_table.fillna(-1)
    gen_table = gen_table.fillna(-1)

    col_names = gt_table.columns.tolist()
    unique_rows = list(gt_table.groupby(col_names).groups)
    ce = 0.

    if not len(gt_caches):
        gt_counts_df = gt_table.groupby(
            col_names).size().reset_index(name='counts')
    gen_counts_df = gen_table.groupby(
        col_names).size().reset_index(name='counts')

    for row in unique_rows:
        value = row
        value_str = ','.join([str(item) for item in row])

        if value_str in gt_caches:
            gt_prob = gt_caches[value_str]
        else:
            gt_prob = gt_counts_df[gt_counts_df[col_names[0]] == value[0]]
            for i in range(len(col_names) - 1):
                gt_prob = gt_prob[gt_prob[col_names[i + 1]] == value[i + 1]]
            if len(gt_prob) == 0:
                print(row)
            #     gt_prob = gt_counts_df[gt_counts_df[col_names[0]] == value[0]]
            #     if len(gt_prob) == 0:
            #         if value[0] != value[0]:
            #             print(value[0])
            #             print(gt_counts_df[gt_counts_df[col_names[0]].isnull()])
            #     print(gt_prob)
            #     for i in range(len(col_names) - 1):
            #         gt_prob = gt_prob[gt_prob[col_names[i + 1]] == value[i + 1]]
            #         print(gt_prob)
            gt_prob = gt_prob.iloc[0]['counts'] / len(gt_table)

            gt_caches[value_str] = gt_prob

        gen_prob = gen_counts_df[gen_counts_df[col_names[0]] == value[0]]
        for i in range(len(col_names) - 1):
            gen_prob = gen_prob[gen_prob[col_names[i + 1]] == value[i + 1]]
        if len(gen_prob) > 0:
            gen_prob = gen_prob.iloc[0]['counts'] / len(gen_table)
        else:
            gen_prob = eps

        ce -= gt_prob * np.log(gen_prob)

    return ce


def relation_generation(sample_frame_dict, join_name_set):
    title_cols = sample_frame_dict['title'].columns.tolist()

    sample_frame_dict['title'] = sample_frame_dict['title'].fillna(-1)

    # generate the other relations from views

    missing_rows = {}
    for col in title_cols:
        missing_rows[col] = []

    union_row_difference = set()
    initial_title_rows = list(
        sample_frame_dict['title'].groupby(title_cols).groups)
    initial_title_rows_str = [
        ','.join([str(item) for item in row]) for row in initial_title_rows]
    for name in join_name_set:
        if name == 'title':
            continue

        # fill nan value with -1 for both data frames
        view_frame = sample_frame_dict[name].fillna(-1)

        all_col = view_frame.columns.tolist()
        print("all columns: {}".format(all_col))
        relation_col = list(set(all_col).difference(title_cols))
        print("columns of the join relation: {}".format(relation_col))
        view_title_rows = list(view_frame.groupby(title_cols).groups)
        view_title_rows_str = [
            ','.join([str(item) for item in row]) for row in view_title_rows]
        row_difference = set(view_title_rows_str).difference(
            set(initial_title_rows_str))
        if row_difference:
            union_row_difference = union_row_difference | row_difference
            print("view {} contain {} rows not found in title".format(
                name, len(row_difference)))

    print("all views together contain {} rows not found in title: {}".format(
        len(union_row_difference), union_row_difference))
    for row_str in union_row_difference:
        row_val = row_str.split(',')
        for i in range(len(title_cols)):
            missing_rows[title_cols[i]].append(row_val[i])

    missing_row_frame = pd.DataFrame.from_dict(missing_rows)
    missing_row_frame = missing_row_frame.astype({'title.kind_id': sample_frame_dict['title'].dtypes[
        'title.kind_id'], 'title.production_year': sample_frame_dict['title'].dtypes['title.production_year']})
    sample_frame_dict['title'] = sample_frame_dict['title'].append(
        missing_row_frame)
    sample_frame_dict['title']['id'] = list(
        range(sample_frame_dict['title'].shape[0]))
    title_frame = sample_frame_dict['title']

    renameDict = {}
    for col in title_cols:
        renameDict[col] = col.split('.')[1]
    print(renameDict)
    title_frame.rename(columns=renameDict, inplace=True)

    title_val_dict = {}
    for row in initial_title_rows_str:
        title_val_dict[row] = []
    for row in union_row_difference:
        title_val_dict[row] = []
    for idx, row in title_frame.iterrows():
        row_str = ",".join([str(row['kind_id']), str(row['production_year'])])
        title_val_dict[row_str].append(idx)

    for name in join_name_set:
        fk_list = []
        if name == 'title':
            continue

        start = time.time()
        # fill nan value with -1 for both data frames
        view_frame = sample_frame_dict[name].fillna(-1)
        for i in range(view_frame.shape[0]):
            row = view_frame.iloc[i]
            row_str = ",".join(
                [str(row['title.kind_id']), str(row['title.production_year'])])
            fk = random.choice(title_val_dict[row_str])
            fk_list.append(fk)

        end = time.time()
        print("Total time takes for assigning foreign keys: {}".format(
            end - start))

        all_cols = view_frame.columns.tolist()
        renameDict = {}
        for col in all_cols:
            renameDict[col] = col.split('.')[1]
        view_frame.rename(columns=renameDict, inplace=True)

        fk_table_name = name.split(',')[0]
        view_frame['movie_id'] = fk_list

        saved_cols = view_frame.columns.tolist()
        saved_cols.remove('kind_id')
        saved_cols.remove('production_year')

        # undo fillna operation
        view_frame = view_frame.applymap(lambda x: np.nan if x == -1 else x)

        view_frame.to_csv(
            './generated_tables_100/{}.csv'.format(fk_table_name), columns=saved_cols, index=False)
        print("Saved {} into csv.".format(fk_table_name))

    # undo fillna operation for title frame
    title_frame = title_frame.applymap(lambda x: np.nan if x == -1 else x)

    print(title_frame.head())

    title_frame.to_csv('./generated_tables_100/title.csv', index=False)
    print("Saved title into csv.")
