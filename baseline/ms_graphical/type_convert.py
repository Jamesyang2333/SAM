import csv
import os
input_dir = "/home_nfs/jingyi/db_generation/db_gen/db_generation_uaeq_nomerge"
output_dir = "/home_nfs/jingyi/db_generation/db_gen/db_generation_uaeq_nomerge_converted"

table_names = ["title", "movie_keyword", "movie_info", "movie_info_idx", "movie_companies", "cast_info"]

suffix = "_200"
for name in table_names:
    rows = []
    input_file = os.path.join(input_dir, name+suffix+".csv")
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first = True
        for row in reader:
            new_row = []
            if first:
                for item in row:
                    new_row.append(item)
                first = False
            else:
                for item in row:
                    if item: 
                        new_row.append(int(float(item)))
                    else:
                        new_row.append(item)
            rows.append(new_row)

    output_file = os.path.join(output_dir, name+suffix+".csv")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)