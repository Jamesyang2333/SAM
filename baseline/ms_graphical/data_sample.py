import csv
import numpy as np
with open(
    "/home_nfs/jingyi/db_generation/queries/mscn_queries_neurocard_format.csv", "r") as f:
    queries = f.readlines()

query_list = []
num_queries = 1000
idx_list = np.random.choice(100000, num_queries, replace=False)

query_list = [queries[idx] for idx in idx_list]

with open('/home_nfs/jingyi/db_generation/queries/mscn_{}.csv'.format(num_queries), 'w', newline='') as file:
    for query in query_list:
        file.write(query)
