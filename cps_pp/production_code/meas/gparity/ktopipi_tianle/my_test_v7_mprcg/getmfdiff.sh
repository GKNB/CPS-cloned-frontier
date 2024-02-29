#!/bin/bash
grep -P "t\s+\d+\s+i" $1 | awk '{print $9}' > $2


