import csv
import errno
import glob
import os

import files as files

listoflists = []
list_of_names= []
path = '/Pesquisa/EP_VDD/*.txt'
files = glob.glob(path)
for name in files:
    path, file_name = os.path.split(name)
    name_replace = file_name.replace('.txt', '')
    list_of_names.append(name_replace)
    try:
        with open(name) as f:
            results = f.read().splitlines(True)
            results.pop(0)
            for row in results:
                replace = row.replace('\t', ',')
                removeBreakline = replace.replace('\n', '')
                vddDoBom = name_replace+','+removeBreakline+',bom'+'\r'
                listoflists.append(vddDoBom)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
with open('quimeras.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(listoflists)