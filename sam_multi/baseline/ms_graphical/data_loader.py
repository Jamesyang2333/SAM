import json
import datasets
from data_utils import MakeTable
import torch
import csv
from torch.utils.data import Dataset, DataLoader



def load_dataset(dataset, file):
    with open(file, 'r', encoding="utf8") as f:
        workload_stats = json.load(f)
    card_list_tmp = workload_stats['card_list']
    query_list = workload_stats['query_list']

    table, _ = MakeTable(dataset)

    card_list_tmp = [float(card)/table.cardinality for card in card_list_tmp]

    columns_list = []
    operators_list = []
    vals_list = []
    card_list = []
    for i, query in enumerate(query_list):
        if card_list_tmp[i] > 0:
            cols = query[0]
            ops = query[1]
            vals = query[2]
            columns_list.append(cols)
            operators_list.append(ops)
            vals_list.append(vals)
            card_list.append(card_list_tmp[i])
    return {"column": columns_list, "operator": operators_list, "val": vals_list, "card": card_list}

class RelationDataset(Dataset):

    def __init__(self, data, label_log, label):
        self.data = data
        self.label_log = label_log
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {"data": self.data[idx], "label_log": self.label_log[idx], "label": self.label[idx]}


def collate_fn(batch):
    batch_data = {}
    batch_data["data"] = torch.FloatTensor([item["data"] for item in batch])
    batch_data["label_log"] = torch.FloatTensor([item["label_log"] for item in batch])
    batch_data["label"] = torch.FloatTensor([item["label"] for item in batch])
    return batch_data

def get_loader(data, label_log, label, batch_size):
    dataset = RelationDataset(data, label_log, label)

    dataLoader = DataLoader(dataset = dataset, batch_size = batch_size, collate_fn = collate_fn)

    return dataLoader


def load_dataset_join(file_name, tables_list, loaded_tables):

    data = []

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
            label = float(row[3])
            current_row["label"] = label
            label_list.append(label)
            data.append(current_row)

    return data