import psycopg2
import numpy as np
from pyspark.sql import SparkSession
from absl import flags
import json

FLAGS = flags.FLAGS

dataset = 'dmv'

s_master = 'local[*]'

def StartSpark(spark=None):
    spark = SparkSession.builder.appName('db_gen')\
        .config('spark.master', s_master)\
        .config('spark.driver.memory', '400g')\
        .config('spark.eventLog.enabled', 'true')\
        .config('spark.sql.warehouse.dir', '/home/ubuntu/spark-sql-warehouse')\
        .config('spark.sql.cbo.enabled', 'true')\
        .config('spark.memory.fraction', '0.9')\
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')\
        .config('spark.memory.offHeap.enabled', 'true')\
        .config('spark.memory.offHeap.size', '100g')\
        .enableHiveSupport()\
        .getOrCreate()

    print('Launched spark:', spark.sparkContext)
    executors = str(
        spark.sparkContext._jsc.sc().getExecutorMemoryStatus().keys().mkString(
            '\n ')).strip()
    print('{} executors:\n'.format(executors.count('\n') + 1), executors)
    return spark

def DropBufferCache():
    worker_addresses = os.path.expanduser('~/hosts-workers')
    if os.path.exists(worker_addresses):
        # If distributed, drop each worker.
        print(
            str(
                subprocess.check_output([
                    'parallel-ssh', '-h', worker_addresses, '--inline-stdout',
                    'sync && sudo bash -c  \'echo 3 > /proc/sys/vm/drop_caches\' && free -h'
                ])))
    else:
        # Drop this machine only.
        subprocess.check_output(['sync'])
       # subprocess.check_output(
        #    ['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'])
        subprocess.check_output(['free', '-h'])

def QueryToPredicate(columns, operators, vals):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    if dataset ==  'dmv':
        count = 0
        for c, v in zip(columns, vals):
            if c == 'Reg Valid Date':
                vals[count] = np.datetime64(v)
            count += 1
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]

    if dataset == 'census':
        count = 0
        for c, v in zip(columns, v_s):
            if c in [0, 4, 10, 11, 12]:
                v_s[count] = str(v)
            else:
                v_s[count] = "\'" + v + "\'"
            count += 1
    elif dataset == 'dmv':
        v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if dataset == 'cencus':
        preds = [
            'c'+str(c) + ' ' + o + ' ' + v
            for c, o, v in zip(columns, operators, v_s)
        ]
    elif dataset == 'dmv':
        preds = [
            c.replace(' ', '_') + ' ' + o + ' ' + v
            for c, o, v in zip(columns, operators, v_s)
        ]
    s = ' and '.join(preds)
    return ' where ' + s


def LoadTable_generated(spark, name, data_dir='./generated_tables/'):
    data_tmp = spark.read.format('csv').option('escape', '\\') \
        .option('header', 'true').option('DELIMITER', ',').csv(data_dir + name)
    data_tmp.createOrReplaceTempView('dmv')

def ExecuteSql(spark, sql):
    df = spark.sql(sql.replace(';', ''))
    return df.collect()

spark = StartSpark()
LoadTable_generated(spark, 'dmv_7_36.csv', './generated_data_tables/')

# load train data

file_str = '../queries/{}workload_with_card_21000.txt'.format(dataset)
with open(file_str, 'r', encoding="utf8") as f:
    workload_stats = json.load(f)

train_card_list = workload_stats['card_list'][0: 7]
train_query_list = workload_stats['query_list'][0: 7]

test_card_list = workload_stats['card_list'][20000: 21000]
test_query_list = workload_stats['query_list'][20000: 21000]

q_error_sum = 0
train_qerror = []
test_qerror = []
spark.catalog.clearCache()
for query, card in zip(train_query_list, train_card_list):
    card = float(card)
    spark.catalog.clearCache()
    pred = QueryToPredicate(query[0], query[1], query[2])
    query_s = 'select count(*) from ' + dataset + pred
    print(query_s)
    result = ExecuteSql(spark, query_s)[0][0]
    if result == 0:
        result = 1
    q_error = max(result/card, card/result)
    print("True cardinality: {}, test cardinality: {}, q error; {}".format(
        card, result, q_error))
    q_error_sum += q_error
    train_qerror.append(q_error)

# for query, card in zip(test_query_list, test_card_list):
#     card = float(card)
#     spark.catalog.clearCache()
#     pred = QueryToPredicate(query[0], query[1], query[2])
#     query_s = 'select count(*) from ' + dataset + pred
#     result = ExecuteSql(spark, query_s)[0][0]
#     if result == 0:
#         result = 1
#     q_error = max(result/card, card/result)
#     print("True cardinality: {}, test cardinality: {}, q error; {}".format(
#         card, result, q_error))
#     q_error_sum += q_error
#     test_qerror.append(q_error)


print("Median: {}".format(np.median(train_qerror)))
print("75th percentile: {}".format(np.percentile(train_qerror, 75)))
print("90th percentile: {}".format(np.percentile(train_qerror, 90)))
print("95th percentile: {}".format(np.percentile(train_qerror, 95)))
print("99th percentile: {}".format(np.percentile(train_qerror, 99)))
print("Max: {}".format(np.max(train_qerror)))
print("Mean: {}".format(np.mean(train_qerror)))


print("Median: {}".format(np.median(test_qerror)))
print("75th percentile: {}".format(np.percentile(test_qerror, 75)))
print("90th percentile: {}".format(np.percentile(test_qerror, 90)))
print("95th percentile: {}".format(np.percentile(test_qerror, 95)))
print("99th percentile: {}".format(np.percentile(test_qerror, 99)))
print("Max: {}".format(np.max(test_qerror)))
print("Mean: {}".format(np.mean(test_qerror)))

