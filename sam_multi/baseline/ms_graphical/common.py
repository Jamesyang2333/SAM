"""Data abstractions."""
import copy
import time

import numpy as np
import pandas as pd
import csv

import torch
from torch.utils import data

import utils
from utils import normalize_data
from torch.utils.data import Dataset, DataLoader


# Na/NaN/NaT Semantics
#
# Some input columns may naturally contain missing values.  These are handled
# by the corresponding numpy/pandas semantics.
#
# Specifically, for any value (e.g., float, int, or np.nan) v:
#
#   np.nan <op> v == False.
#
# This means that in progressive sampling, if a column's domain contains np.nan
# (at the first position in the domain), it will never be a valid sample
# target.
#
# The above evaluation is consistent with SQL semantics.


class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('Attr1').Fill(data, infer_dist=True)

    The passed-in 'data' is copied by reference.
    """

    def __init__(self, name, distribution_size=None, pg_name=None):
        self.name = name

        # Data related fields.
        self.data = None
        self.all_distinct_values = None
        self.all_distinct_values_with_counts = None
        self.distribution_size = distribution_size

        # pg_name is the name of the corresponding column in Postgres.  This is
        # put here since, e.g., PG disallows whitespaces in names.
        self.pg_name = pg_name if pg_name else name

    def Name(self):
        """Name of this column."""
        return self.name

    def DistributionSize(self):
        """This column will take on discrete values in [0, N).

        Used to dictionary-encode values to this discretized range.
        """
        return self.distribution_size

    def ValToBin(self, val):
        if isinstance(self.all_distinct_values, list):
            return self.all_distinct_values.index(val)
        inds = np.where(self.all_distinct_values == val)
        assert len(inds[0]) > 0, val

        return inds[0][0]

    def SetDistribution(self, distinct_values, distinct_values_with_counts):
        """This is all the values this column will ever see."""
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(distinct_values)
        contains_nan = np.any(is_nan)
        dv_no_nan = distinct_values[~is_nan]
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(dv_no_nan))
        if contains_nan and np.issubdtype(distinct_values.dtype, np.datetime64):
            vs = np.insert(vs, 0, np.datetime64('NaT'))
        elif contains_nan:
            vs = np.insert(vs, 0, np.nan)
        if self.distribution_size is not None:
            assert len(vs) == self.distribution_size
        self.all_distinct_values = vs
        self.distribution_size = len(vs)
        self.all_distinct_values_with_counts = distinct_values_with_counts.sort_index()
        return self

    def Fill(self, data_instance, infer_dist=False):
        assert self.data is None
        self.data = data_instance
        # If no distribution is currently specified, then infer distinct values
        # from data.
        if infer_dist:
            self.SetDistribution(self.data, None)
        return self

    def __repr__(self):
        return 'Column({}, distribution_size={})'.format(
            self.name, self.distribution_size)


class Table(object):
    """A collection of Columns."""

    def __init__(self, name, columns, pg_name=None):
        """Creates a Table.

        Args:
            name: Name of this table object.
            columns: List of Column instances to populate this table.
            pg_name: name of the corresponding table in Postgres.
        """
        self.name = name
        self.cardinality = self._validate_cardinality(columns)
        self.columns = columns

        self.val_to_bin_funcs = [c.ValToBin for c in columns]
        self.name_to_index = {c.Name(): i for i, c in enumerate(self.columns)}

        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name

    def __repr__(self):
        return '{}({})'.format(self.name, self.columns)

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [len(c.data) for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def Name(self):
        """Name of this table."""
        return self.name

    def Columns(self):
        """Return the list of Columns under this table."""
        return self.columns

    def ColumnIndex(self, name):
        """Returns index of column with the specified name."""
        assert name in self.name_to_index, (self.name, name,
                                            list(self.name_to_index.keys()))
        return self.name_to_index[name]


class CsvTable(Table):
    """Wraps a CSV file or pd.DataFrame as a Table."""

    def __init__(self,
                 name,
                 filename_or_df,
                 cols,
                 type_casts={},
                 pg_name=None,
                 pg_cols=None,
                 **kwargs):
        """Accepts the same arguments as pd.read_csv().

        Args:
            filename_or_df: pass in str to reload; otherwise accepts a loaded
              pd.Dataframe.
            cols: list of column names to load; can be a subset of all columns.
            type_casts: optional, dict mapping column name to the desired numpy
              datatype.
            pg_name: optional str, a convenient field for specifying what name
              this table holds in a Postgres database.
            pg_cols: optional list of str, a convenient field for specifying
              what names this table's columns hold in a Postgres database.
            **kwargs: keyword arguments that will be pass to pd.read_csv().
        """
        self.name = name
        self.pg_name = pg_name

        if isinstance(filename_or_df, str):
            self.data = self._load(filename_or_df, cols, **kwargs)
        else:
            assert (isinstance(filename_or_df, pd.DataFrame))
            self.data = filename_or_df

        self.columns = self._build_columns(self.data, cols, type_casts, pg_cols)

        self.data_cardinality_list = []
        self._buil_cardinality_list(self.data)

        super(CsvTable, self).__init__(name, self.columns, pg_name)

    def _load(self, filename, cols, **kwargs):
        print('Loading csv...', end=' ')
        s = time.time()
        # df = pd.read_csv(filename, usecols=cols, **kwargs)
        df = pd.read_csv(filename, header=0, names=cols, **kwargs)
        if cols is not None:
            df = df[cols]
        print('done, took {:.1f}s'.format(time.time() - s))
        print(df.head())
        return df

    def _build_columns(self, data, cols, type_casts, pg_cols):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print('Parsing...', end=' ')
        s = time.time()
        for col, typ in type_casts.items():
            if col not in data:
                continue
            if typ != np.datetime64:
                data[col] = data[col].astype(typ, copy=False)
            else:
                # Both infer_datetime_format and cache are critical for perf.
                data[col] = pd.to_datetime(data[col],
                                           infer_datetime_format=True,
                                           cache=True)

        # Discretize & create Columns.
        if cols is None:
            cols = data.columns
        columns = []
        if pg_cols is None:
            pg_cols = [None] * len(cols)
        for c, p in zip(cols, pg_cols):
            col = Column(c, pg_name=p)
            col.Fill(data[c])

            # dropna=False so that if NA/NaN is present in data,
            # all_distinct_values will capture it.
            #
            # For numeric: np.nan
            # For datetime: np.datetime64('NaT')
            col.SetDistribution(data[c].value_counts(dropna=False).index.values, data[c].value_counts(dropna=False))
            columns.append(col)
        print('done, took {:.1f}s'.format(time.time() - s))
        return columns

    def _buil_cardinality_list(self, data):
        data_cardinality_list = data.groupby(data.columns.tolist())[data.columns.tolist()[0]].transform('size')
        self.data_cardinality_list = (np.array(data_cardinality_list)/len(self.columns[0].data))

class TableDataset(data.Dataset):
    """Wraps a Table and yields each row as a PyTorch Dataset element."""

    def __init__(self, table):
        super(TableDataset, self).__init__()
        self.table = copy.deepcopy(table)

        print('Discretizing table...', end=' ')
        s = time.time()
        # [cardianlity, num cols].
        self.tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        self.tuples = torch.as_tensor(
            self.tuples_np.astype(np.float32, copy=False))
        print('done, took {:.1f}s'.format(time.time() - s))

    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return Discretize(col)

    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return (idx, self.tuples[idx])


def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids

class QueryDataset(data.Dataset):
    """Wraps a Table and yields each row as a PyTorch Dataset element."""

    def __init__(self, valid_i_list):
        super(QueryDataset, self).__init__()
        self.valid_i_list = copy.deepcopy(valid_i_list)


        print('Handling queries...', end=' ')
        s = time.time()
        print('done, took {:.1f}s'.format(time.time() - s))

    def size(self):
        return len(self.valid_i_list)

    def __len__(self):
        return len(self.valid_i_list)

    def __getitem__(self, idx):
        return (idx, self.valid_i_list[idx])

class JoinQueryDataset(data.Dataset):

    def __init__(self, file_name, tables_list, loaded_tables, cardinality):
        super(JoinQueryDataset, self).__init__()

        self.data = []

        columns_list = []
        operators_list = []
        vals_list = []
        label_list = []

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='#')
            line_count = 0
            for row in csv_reader:
                current_row = {}
                tables = []
                tables = row[0].split(',')
                tables = [table.split(' ')[1] for table in tables]
                current_row["tables"] = tables
                join_con = row[1]
                current_row["join_con"] = join_con
                predicates = row[2].split(',')
                cols = []
                operators = []
                vals = []
                for i in range(len(predicates)//3):
                    cols.append(predicates[i*3])
                    operators.append(predicates[i*3+1])
                    vals.append(predicates[i*3+2])

                columns_list.append(cols)
                operators_list.append(operators)
                vals_list.append(vals)

                current_row["predicates"] = [cols, operators, vals]
                label = float(row[3]) / cardinality
                current_row["labels"] = label
                label_list.append(label)
                self.data.append(current_row)
        
        initial_range_dict = {}
        column_min_max_vals_dict = {}
        for i in range(len(tables_list)):

            table = loaded_tables[tables_list[i]]
            column_min_max_vals = {}
            num_cols = len(table.columns)
            initial_range = []
            for col_idx in range(num_cols):
                column = table.columns[col_idx]
                all_col_vals = column.all_distinct_values

                col_min_val = 0
                col_max_val = len(all_col_vals) - 1


                # if table_name in ['dmv', 'dmv-tiny'] and col_idx == 6:
                #     print(all_col_vals)
                #     col_min_val = all_col_vals[0] if all_col_vals[0] != 0 else all_col_vals[1]
                #     col_max_val = all_col_vals[-1]
                # elif table_name == 'adult' and col_idx in [0, 3, 9, 10, 11]:
                #     col_min_val = all_col_vals[0]
                #     col_max_val = all_col_vals[-1]
                # else:
                #     col_min_val = 0
                #     col_max_val = len(all_col_vals) - 1
                column_min_max_vals[column.name] = [float(col_min_val), float(col_max_val)]
                initial_range.append(np.array([float(0.), float(1.)]))

            column_min_max_vals_dict[tables_list[i]] = column_min_max_vals
            initial_range_dict[tables_list[i]] = initial_range


        vals_list = discretize_vals(columns_list, vals_list, tables_list, loaded_tables)
        self.table_features_list = []
        l_len = len(columns_list)
        for i in range(l_len):
            table_features = []
            tables = self.data[i]["tables"]
            if i == 0:
                print(self.data[i]["tables"])
            for table in self.data[i]["tables"]:
                table_features.append(copy.deepcopy(initial_range_dict[table]))
            cols = columns_list[i]
            ops = operators_list[i]
            vals = vals_list[i]
            q_len = len(cols)
            for j in range(q_len):
                table_name, col_name = cols[j].split('.')
                table = loaded_tables[table_name]
                col_id = table.ColumnIndex(col_name)
                op = ops[j]
                val = vals[j]
                norm_val = normalize_data(val, col_name, column_min_max_vals_dict[table_name])
                if op == '=':
                    table_features[tables.index(table_name)][col_id][0] = norm_val
                    table_features[tables.index(table_name)][col_id][1] = norm_val
                elif op in ['<', '<=']:
                    table_features[tables.index(table_name)][col_id][1] = norm_val
                else:
                    table_features[tables.index(table_name)][col_id][0] = norm_val

            if i == 0:
                print(table_features)

            for j in range(len(table_features)):
                # print(table_features[j])
                table_features[j] = np.concatenate(table_features[j])
            
            self.table_features_list.append(table_features)
        
        # normalize label list
        label_norm, min_val, max_val, is_add_eps = utils.normalize_labels(label_list)
        self.min_val = min_val
        self.max_val = max_val
        self.is_add_eps = is_add_eps
        self.label_norm = label_norm
        self.label_list = label_list

    def size(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"info": self.data[idx],  "feature": self.table_features_list[idx], "label_log": self.label_norm[idx], "label": self.label_list[idx]}

def discretize_vals(columns_list, vals_list, tables_list, loaded_tables):
    res_vals_list = []
    list_len = len(columns_list)
    for i in range(list_len):
        temp = []
        cols = columns_list[i]
        vals = vals_list[i]
        q_len = len(cols)
        for j in range(q_len):
            table_name, col_name = cols[j].split('.')
            table = loaded_tables[table_name]
            val = vals[j]
            col_idx = table.ColumnIndex(col_name)
            column = table.columns[col_idx]
            if isinstance(type(val), type(column.all_distinct_values[-1])) is False:
                val = type(column.all_distinct_values[-1])(val)
            if isinstance(type(column.all_distinct_values[0]),
                            type(column.all_distinct_values[-1])) is False:
                column.all_distinct_values[0] = type(
                    column.all_distinct_values[-1])(
                    column.all_distinct_values[0])
            if not (val in column.all_distinct_values):
                print(val)
                print(table_name)
                print(col_name)
                print(column.all_distinct_values)
            temp.append(float(list(column.all_distinct_values).index(val)))
            # if table_name in ['dmv', 'dmv-tiny'] and col_idx == 6:
            #     if val != np.datetime64('NaT'):
            #         temp.append(float(np.datetime64(val)))
            #     else:
            #         temp.append(0)
            # elif table_name == 'adult' and col_idx in [0,3,9,10,11]:
            #     temp.append(float(val))
            # else:
            #     column = table.columns[col_idx]
            #     if isinstance(type(val), type(column.all_distinct_values[-1])) is False:
            #         val = type(column.all_distinct_values[-1])(val)
            #     if isinstance(type(column.all_distinct_values[0]),
            #                   type(column.all_distinct_values[-1])) is False:
            #         column.all_distinct_values[0] = type(
            #             column.all_distinct_values[-1])(
            #             column.all_distinct_values[0])

            #     temp.append(float(list(column.all_distinct_values).index(val)))
        res_vals_list.append(temp)
    return res_vals_list

def collate_fn(batch):
    batch_data = {}
    batch_data["info"] = ([item["info"] for item in batch])
    batch_data["feature"] = [[torch.FloatTensor(feature) for feature in item["feature"]] for item in batch]
    batch_data["label_log"] = torch.FloatTensor([item["label_log"] for item in batch])
    batch_data["label"] = torch.FloatTensor([item["label"] for item in batch])
    return batch_data

def get_loader(queryDataset, batch_size):

    dataLoader = DataLoader(dataset = queryDataset, batch_size = batch_size, collate_fn = collate_fn)

    return dataLoader


class JoinQueryDataset(data.Dataset):

    def __init__(self, file_name, tables_list, loaded_tables, cardinality):
        super(JoinQueryDatasetNew, self).__init__()

        self.data = []

        columns_list = []
        operators_list = []
        vals_list = []
        label_list = []

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='#')
            line_count = 0
            for row in csv_reader:
                current_row = {}
                tables = []
                tables = row[0].split(',')
                # tables = [table.split(' ')[1] for table in tables]
                current_row["tables"] = tables
                join_con = row[1]
                current_row["join_con"] = join_con
                predicates = row[2].split(',')
                cols = []
                operators = []
                vals = []
                for i in range(len(predicates)//3):
                    cols.append(predicates[i*3])
                    operators.append(predicates[i*3+1])
                    # get rid of quote if exist
                    vals.append(predicates[i*3+2].strip("'"))

                columns_list.append(cols)
                operators_list.append(operators)
                vals_list.append(vals)

                current_row["predicates"] = [cols, operators, vals]
                label = float(row[3]) / cardinality
                current_row["labels"] = label
                label_list.append(label)
                self.data.append(current_row)
        
        initial_range_dict = {}
        column_min_max_vals_dict = {}
        for i in range(len(tables_list)):

            table = loaded_tables[tables_list[i]]
            column_min_max_vals = {}
            num_cols = len(table.columns)
            initial_range = []
            for col_idx in range(num_cols):
                column = table.columns[col_idx]
                all_col_vals = column.all_distinct_values

                col_min_val = 0
                col_max_val = len(all_col_vals) - 1


                # if table_name in ['dmv', 'dmv-tiny'] and col_idx == 6:
                #     print(all_col_vals)
                #     col_min_val = all_col_vals[0] if all_col_vals[0] != 0 else all_col_vals[1]
                #     col_max_val = all_col_vals[-1]
                # elif table_name == 'adult' and col_idx in [0, 3, 9, 10, 11]:
                #     col_min_val = all_col_vals[0]
                #     col_max_val = all_col_vals[-1]
                # else:
                #     col_min_val = 0
                #     col_max_val = len(all_col_vals) - 1
                column_min_max_vals[column.name] = [float(col_min_val), float(col_max_val)]
                initial_range.append(np.array([float(0.), float(1.)]))

            column_min_max_vals_dict[tables_list[i]] = column_min_max_vals
            initial_range_dict[tables_list[i]] = initial_range


        vals_list = discretize_vals(columns_list, vals_list, tables_list, loaded_tables)
        self.table_features_list = []
        l_len = len(columns_list)
        for i in range(l_len):
            table_features = []
            tables = self.data[i]["tables"]
            if i == 0:
                print(self.data[i]["tables"])
            for table in self.data[i]["tables"]:
                table_features.append(copy.deepcopy(initial_range_dict[table]))
            cols = columns_list[i]
            ops = operators_list[i]
            vals = vals_list[i]
            q_len = len(cols)
            for j in range(q_len):
                table_name, col_name = cols[j].split('.')
                table = loaded_tables[table_name]
                col_id = table.ColumnIndex(col_name)
                op = ops[j]
                val = vals[j]
                norm_val = normalize_data(val, col_name, column_min_max_vals_dict[table_name])
                if op == '=':
                    table_features[tables.index(table_name)][col_id][0] = norm_val
                    table_features[tables.index(table_name)][col_id][1] = norm_val
                elif op in ['<', '<=']:
                    table_features[tables.index(table_name)][col_id][1] = norm_val
                else:
                    table_features[tables.index(table_name)][col_id][0] = norm_val

            if i == 0:
                print(table_features)

            for j in range(len(table_features)):
                # print(table_features[j])
                table_features[j] = np.concatenate(table_features[j])
            
            self.table_features_list.append(table_features)
        
        # normalize label list
        label_norm, min_val, max_val, is_add_eps = utils.normalize_labels(label_list)
        self.min_val = min_val
        self.max_val = max_val
        self.is_add_eps = is_add_eps
        self.label_norm = label_norm
        self.label_list = label_list

    def size(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"info": self.data[idx],  "feature": self.table_features_list[idx], "label_log": self.label_norm[idx], "label": self.label_list[idx]}