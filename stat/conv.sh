#!/bin/sh

TMPFILE=tmp
head -n 1 ${1} > ${TMPFILE}
sed -n '2 p' < ${1} | awk '{print $1,$2,$3,90,90,90}'  >> ${TMPFILE}
sed -n '3,$p' < ${1} | awk '{print $1,$2,$3,$4}' | sed 's/^0/Ge/'  | sed 's/^1/Se/' | sed 's/^2/Te/'  >> ${TMPFILE}
mv -v ${TMPFILE} ${1}
