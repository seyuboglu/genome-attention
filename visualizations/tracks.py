from matplotlib import pyplot as plt
import numpy as np
import pyBigWig
import collections

bw_pred = pyBigWig.open("/data/gangus/restore/dilation_multi_1_5/test_out/tracks/t0_preds.bw")
bw_true = pyBigWig.open("/data/gangus/restore/dilation_multi_1_5/test_out/tracks/t0_true.bw")

def detect_breaks(bw):
    intervals = bw.intervals('chr2')
    num_failures = 0
    for i in range(len(intervals) - 1):
        prev = intervals[i][1]
        curr = intervals[i+1][0]
        if prev != curr:
            print('---- failed ----')
            print(i)
            print(intervals[i])
            print(intervals[i+1])
            print('----------------')
            num_failures += 1

def detect_same_intervals(bw1, bw2):
    intervals1 = bw1.intervals('chr2')
    intervals2 = bw2.intervals('chr2')
    # num breaks
    for i in range(4):
        print('i = {}'.format(i))
        print(intervals1[i * 960])
        print(intervals1[((i+1) * 960) - 1])
        print(intervals2[(i * 1024) + 32])
        print(intervals2[((i+1) * 1024) - 1 - 32])
        print('--')

# detect_breaks(bw_true)
# exit()
detect_same_intervals(bw_pred, bw_true)
exit()



iters = 0

types = []
for line in open("/data/genome-attention/l131k_w128.bed"):
    cols = line.strip().split()
    print(cols)
    break
    types.append(cols[3])
    vals = (cols[0], int(cols[1]), int(cols[2]))
    # print(vals)
    # if cols[3] == 'test':

    iters += 1
    # if iters ==20:
    #     break
    # Do something with the values...

print(iters)
print(collections.Counter(types))
bw.close()