import psycopg2
import numpy as np

import time

conn = psycopg2.connect(
    host="localhost",
    database="db_gen_pgm_170",
    user="jingyi",
    port="5444"
)

cur = conn.cursor()

queries = open(
    "/home_nfs/jingyi/db_generation/queries/job-light.sql", "r")
cards = open(
    "/home_nfs/jingyi/db_generation/queries/job-light-card.csv")

card_list = []
for line in cards:
    card_list.append(int(float(line.strip())))

query_list = []
for line in queries:
    query_list.append(line.strip())

start = time.time()

# n_test_queries = min(100, len(query_list))
n_test_queries = len(query_list)
# n_test_queries = 100
q_error_list = []
result_list = []
for i in range(n_test_queries):
    cur.execute(query_list[i])
    result = cur.fetchone()[0]
    if result == 0:
        result = 1
    q_error = max(result/card_list[i], card_list[i]/result)
    print("True cardinality: {}, test cardinality: {}, q error; {}".format(
        card_list[i], result, q_error))
    q_error_list.append(q_error)
    result_list.append(result)

end = time.time()
print("Total time takes for executing {} queries: {}".format(
        n_test_queries, end - start))

q_error_list = np.array(q_error_list)
result_list = np.array(result_list)
np.savetxt('./result/gen_train_500_100_query_100.csv', result_list)
print("Max q error: {}".format(np.max(q_error_list)))
print("99 percentile q error: {}".format(np.percentile(q_error_list, 99)))
print("95 percentile q error: {}".format(np.percentile(q_error_list, 95)))
print("90 percentile q error: {}".format(np.percentile(q_error_list, 90)))
print("75 percentile q error: {}".format(np.percentile(q_error_list, 75)))
print("50 percentile q error: {}".format(np.percentile(q_error_list, 50)))
print("Average q error: {}".format(np.mean(q_error_list)))
