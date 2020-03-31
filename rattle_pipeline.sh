#!/bin/bash

indir=$1
infq=$2
subn=$3
outdir=$4
ksize=$5
ncore=$6

# mk output dir
mkdir ${outdir} && \

# subsample
seqtk sample -s100 \
${indir}/${infq} \
${subn} > ${indir}/subsample.fq && \

# rattle
docker run --rm \
-v /mnt/ibm_lg/daniel_le:/mnt/ibm_lg/daniel_le \
-u $(id -u):$(id -g) \
rattle \
/rattle_module.sh \
${indir}/subsample.fq \
${outdir} \
10 \
${ncore} && \

# map untrimmed
docker run --rm \
-v /mnt/ibm_lg/daniel_le:/mnt/ibm_lg/daniel_le \
-u $(id -u):$(id -g) \
minimap2 \
/map.sh \
${indir}/${infq} \
${outdir}/transcriptome.fa \
${outdir} \
${ncore} && \

# mk trim subdir
mkdir ${outdir}/trim && \

# trim
python3 /home/daniel_le/git/darmanis_group_DL_notebook/trim_fa.py \
${outdir}/transcriptome.fa \
${outdir}/mapped.sorted.bam \
${outdir}/trim/transcriptome_trimmed.fa \
${ncore} && \

# map trimmed
docker run --rm \
-v /mnt/ibm_lg/daniel_le:/mnt/ibm_lg/daniel_le \
-u $(id -u):$(id -g) \
minimap2 \
/map.sh \
${indir}/${infq} \
${outdir}/trim/transcriptome_trimmed.fa \
${outdir}/trim \
${ncore}