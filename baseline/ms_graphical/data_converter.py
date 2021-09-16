import csv
queries = open(
    "/home_nfs/jingyi/db_generation/queries/mscn_1000.csv", "r")
card_list = []
query_list = []

for query in queries:
    elements = query.split("#")
    tables = elements[0].split(',')
    # print(tables)
    condition_list = []
    if elements[1]:
        join_conditions = elements[1].split(',')
        for condition in join_conditions:
            condition_list.append(condition)

    filter_conditions = elements[2].split(',')
    n_filter_conditions = len(filter_conditions) // 3
    for i in range(n_filter_conditions):
        condition_list.append(
            filter_conditions[i*3]+filter_conditions[i*3+1]+filter_conditions[i*3+2])

    # print(join_list)
    # print(filter_list)
    tables.sort()
    query_list.append((tables, condition_list))
    card_list.append(int(elements[3]))


with open('/home_nfs/jingyi/db_generation/queries/mscn_1000.sql', 'w', newline='') as file:
    for i in range(len(query_list)):
        tables = query_list[i][0]
        conditions = query_list[i][1]
        file.write("SELECT COUNT(*) FROM " + ','.join(tables) +
                   ' WHERE ' + ' AND '.join(conditions) + '\n')

with open('/home_nfs/jingyi/db_generation/queries/mscn_1000_card.csv', 'w', newline='') as file:
    for i in range(len(card_list)):
        file.write(str(card_list[i]) + '\n')
