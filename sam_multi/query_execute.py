import psycopg2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--queries',
                    default="./queries/mscn_400.sql",
                    type=str,
                    required=True,
                    help='Test query file')
parser.add_argument('--cards',
                    default="./queries/mscn_400_card.csv",
                    type=str,
                    required=True,
                    help='Test querycardinality file')
args = parser.parse_args()

conn = psycopg2.connect(
    host="localhost",
    database="db_gen_uaeq_400_test_100",
    user="jingyi",
    port="5444"
)

cur = conn.cursor()

queries = open(
    args.queries, "r")
cards = open(
    args.cards, "r")

card_list = []
for line in cards:
    card_list.append(int(float(line.strip())))

query_list = []
for line in queries:
    query_list.append(line.strip())

start = time.time()

n_test_queries = len(query_list)
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
# np.savetxt('./result/gen_train_500_100_query_100.csv', result_list)
print("Max q error: {}".format(np.max(q_error_list)))
print("99 percentile q error: {}".format(np.percentile(q_error_list, 99)))
print("95 percentile q error: {}".format(np.percentile(q_error_list, 95)))
print("90 percentile q error: {}".format(np.percentile(q_error_list, 90)))
print("75 percentile q error: {}".format(np.percentile(q_error_list, 75)))
print("50 percentile q error: {}".format(np.percentile(q_error_list, 50)))
print("Average q error: {}".format(np.mean(q_error_list)))
