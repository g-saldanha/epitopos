import csv
import errno
import glob
import os

path = 'D:/Gabriel/Downloads/candidate.csv'
files = glob.glob(path)
for name in files:
    # path, file_name = os.path.split(name)
    # name_replace = file_name.replace('.txt', '')
    try:
        with open(name) as f:
            results = f.read().splitlines(True)
            results.pop(0)
            for row in results:
                replace = row.replace('\t', ',')
                removeBreakline = replace.replace('\n', '')
                vddDoBom = name_replace+','+removeBreakline+',quimeras'+'\r'
                listoflists.append(vddDoBom)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
with open('quimeras.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(listoflists)