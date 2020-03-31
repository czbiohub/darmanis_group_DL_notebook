#!/bin/bash

INPUT_BAM=$1
CHROM="chr9"
START=7538434
END=7545000

#samtools mpileup -uf REFERENCE.fasta SAMPLE.bam | bcftools call -c | vcfutils.pl vcf2fq > SAMPLE_cns.fastq

# slice on region
# samtools -h view ${INPUT_BAM} ${CHROM}:${START}-${END} | \
# read starts or stops in region 
# awk '$1 ~ /^@/ || s <= $4 <= e && s <= $8 <= e {print $0}' s="${START}" e="${END}" | \
# samtools pileup
samtools mpileup -uf /mnt/ibm_lg/daniel_le/data/botryllus/genome/botznik-chr.fa -r "${CHROM}:${START}-${END}" $INPUT_BAM | \
# variant call
bcftools call -c | \
# awk 's <= $2 <= e {print $0}' s="${START}" e="${END}" | \
# cns caller
vcfutils.pl vcf2fq > cns.fastq