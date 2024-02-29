#!/bin/bash
cat compare_$1.log | grep 'reldiff' | awk '{print $5}' > diffs_$1.dat
