#!/bin/bash
sed -i.sav -f misc/scripts/del_cvs_tag.sed $1
vi $1
