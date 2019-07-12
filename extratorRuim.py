import csv
import errno
import glob
import os

listoflists = []
path = 'D:/Gabriel/Docs/Pesquisa/n_ep/*.txt'
files = glob.glob(path)
for name in files:
    path, file_name = os.path.split(name)
    name_replace = file_name.replace('.txt', '')
    try:
        with open(name) as f:
            results = f.read().splitlines(True)
            results.pop(0)
            for row in results:
                if not 'Antigenicidade' in row:
                    replace = row.replace('\t', ',')
                    removeBreakline = replace.replace('\n', '')
                    vddDoBom = name_replace+','+removeBreakline+',ruim'+'\r'
                    listoflists.append(vddDoBom)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
with open('n_ep.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(listoflists)