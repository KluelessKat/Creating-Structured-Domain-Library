#!/usr/bin/python3

import iupred3a_lib
import os
import sys

help_msg = """Usage: {} (options) (seqfile)

Options
\t-d str   -   Location of data directory (default='./')""".format(sys.argv[0])
if len(sys.argv) < 2:
    sys.exit(help_msg)
if not os.path.isfile(sys.argv[-1]):
    sys.exit('Input sequence file not found at {}!\n{}'.format(sys.argv[-2], help_msg))
if '-d' in sys.argv:
    PATH = sys.argv[sys.argv.index('-d') + 1]
    if not os.path.isdir(os.path.join(PATH, 'data')):
        sys.exit('Data directory not found at {}!\n{}'.format(PATH, help_msg))


PATH = os.path.dirname(os.path.realpath(__file__))
seq = iupred3a_lib.read_seq(sys.argv[-1])
iup = iupred3a_lib.iupred(seq, new_smoothing=False)[0]
red = iupred3a_lib.iupred_redox(seq, new_smoothing=False)[0]
regions = iupred3a_lib.get_redox_regions(red, iup)
if not os.path.isdir(PATH):
    sys.exit('Data directory not found at {}!\n{}'.format(PATH, help_msg))
if '-d' in sys.argv:
    PATH = sys.argv[sys.argv.index('-d') + 1]
    if not os.path.isdir(os.path.join(PATH, 'data')):
        sys.exit('Data directory not found at {}!\n{}'.format(PATH, help_msg))
print("""# Large-scale analysis of redox-sensitive conditionally disordered protein regions reveals their widespread nature and key roles in high-level eukaryotic processes
# Gabor Erdos, Balint Meszaros, Dana Reichmann, Zsuzsanna Dosztanyi
# PROTEOMICS 2018, Submitted
#
# Prediction output
# POS\tRES\tIUPRED2\tREDOX\tREGION""")
for idx in range(len(seq)):
    reg = 0
    for start, end in regions.items():
        if start <= idx <= end - 1:
            reg = 1
            break
    print("{}\t{}\t{:.2f}\t{:.2f}\t{}".format(idx+1, seq[idx], iup[idx], red[idx], reg))
