import csv
import errno
import glob
import os

import files as files

listoflists = []
list_of_names= []
path1 = './Pesquisa/EP_VDD/*.txt'
path2 = './Pesquisa/n_ep/*.txt'
files1 = glob.glob(path1)
files2 = glob.glob(path2)
identificador = 2000

for name in files1:
    path, file_name = os.path.split(name)
    name_replace = file_name.replace('.txt', '')
    list_of_names.append(str(identificador)+','+str(name_replace)+'\r')
    try:
        with open(name) as f:
            results = f.read().splitlines(True)
            results.pop(0)
            for row in results:
                replace = row.replace('\t', ',')
                removeBreakline = replace.replace('\n', '')
                vddDoBom = str(identificador)+','+removeBreakline+',bom'+'\r'
                listoflists.append(vddDoBom)
        identificador = identificador + 1
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

for name in files2:
    path, file_name = os.path.split(name)
    name_replace = file_name.replace('.txt', '')
    list_of_names.append(str(identificador)+','+str(name_replace)+'\r')
    try:
        with open(name) as f:
            results = f.read().splitlines(True)
            results.pop(0)
            for row in results:
                if not 'Antigenicidade' in row:
                    replace = row.replace('\t', ',')
                    removeBreakline = replace.replace('\n', '')
                    vddDoBom = str(identificador)+','+removeBreakline+',ruim'+'\r'
                    listoflists.append(vddDoBom)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
with open('epitopos.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(listoflists)

with open('tab_dominio.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(list_of_names)