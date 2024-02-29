#!/bin/bash

rm compare_0.log
rm diffs_0.dat
./compare_all.pl 0 > compare_0.log
./getdiffs.sh 0
echo "Main diff hist:"
python gen_hist.py diffs_0.dat

grep "not exist" compare_0.log

#rm mfcompare_0.log
#rm mfdiffs_0.dat
#./mf_compare_all.pl 0 > mfcompare_0.log
#./getmfdiff.sh mfcompare_0.log mfdiffs_0.dat

#echo "MF diff hist:"
#python gen_hist.py mfdiffs_0.dat
