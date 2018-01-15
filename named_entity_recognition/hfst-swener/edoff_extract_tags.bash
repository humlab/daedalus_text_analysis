#!/bin/bash
OPTIND=1
year_offset=0
while getopts "b:" opt; do
	case "$opt" in
	b) year_offset=$OPTARG
	   ;;
	esac
done
shift $((OPTIND-1))
source_files=$@
for fullfile in $source_files
do
	filename="${fullfile##*/}"
	/usr/local/bin/extract-tagged-names.py $fullfile | nawk -v COLUMN1=$filename '{printf "%s\t%s\r\n", COLUMN1, $0}'
done
exit 1
