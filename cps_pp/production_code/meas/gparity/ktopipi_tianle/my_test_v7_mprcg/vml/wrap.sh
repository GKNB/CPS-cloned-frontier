#!/bin/sh
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

$@