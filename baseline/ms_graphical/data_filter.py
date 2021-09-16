import csv
queries = open(
    "/home_nfs/jingyi/db_generation/queries/mscn_queries_neurocard_format.csv", "r")

query_list = []
count = 0
num_queries = 180
total_count = 0

for query in queries:
    total_count += 1
    elements = query.split("#")
    tables = elements[0].split(',')

    if (len(tables) == 1 and tables[0] == 'title') or len(tables) == 2:
        count += 1
    
    query_list.append(query)
    if count == num_queries:
        break

print(count)
print(total_count)
with open('/home_nfs/jingyi/db_generation/queries/mscn_{}.csv'.format(num_queries), 'w', newline='') as file:
    for query in query_list:
        file.write(query)
