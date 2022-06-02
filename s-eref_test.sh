#!/bin/bash

## early remember early forgetting
## nohup sh ./sparse_models/deit-s-eref_test > logs/s-eref-bs128-test.log 2>&1 &

BMDIR=${1}
mkdir -p ${BMDIR}

## pre_sparse
mkdir -p ${BMDIR}/ps
touch ${BMDIR}/ps/ps_attn_bm_out.txt
touch ${BMDIR}/ps/ps_mlp_bm_out.txt
touch ${BMDIR}/ps/ps_patch_bm_out.txt

CUDA_VISIBLE_DEVICES=3 python3 
-m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM \--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 128 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/ps \
--pretrained_path pretrained_models/deit_small_patch16_224-cd65a155.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 1 -- stage pre_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_fileout ${BMDIR}/ps/ps_attn_bm_out.txt \
--mlp_bp_methods_fileout ${BMDIR}/ps/ps_mlp_bm_out.txt \
--patch_bp_methods_fileout ${BMDIR}/ps/ps_patch_bm_out.txt

echo "PRE SPARSE completed."


## middle_mix
mkdir -p ${BMDIR}/mm
touch ${BMDIR}/mm/mm_attn_bm_out.txt
touch ${BMDIR}/mm/mm_mlp_bm_out.txt
touch ${BMDIR}/mm/mm_patch_bm_out.txt

CUDA_VISIBLE_DEVICES=3 python3 
-m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM \--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 128 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/mm \
--pretrained_path ${BMDIR}/ps/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 2 -- stage pre_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/ps/ps_attn_bm_out.txt \
--mlp_bp_methods_filein ${BMDIR}/ps/ps_mlp_bm_out.txt \
--patch_bp_methods_filein ${BMDIR}/ps/ps_patch_bm_out.txt \
--attn_bp_methods_fileout ${BMDIR}/mm/mm_attn_bm_out.txt \
--mlp_bp_methods_fileout ${BMDIR}/mm_mlp_bm_out.txt \
--patch_bp_methods_fileout ${BMDIR}/mm/mm_patch_bm_out.txt

echo "MIDDLE MIX completed."


## middle_sparse
mkdir -p ${BMDIR}/ms
touch ${BMDIR}/ms/ms_attn_bm_out.txt
touch ${BMDIR}/ms/ms_mlp_bm_out.txt
touch ${BMDIR}/ms/ms_patch_bm_out.txt

CUDA_VISIBLE_DEVICES=3 python3 
-m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM \--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 128 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/ms \
--pretrained_path ${BMDIR}/mm/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 2 -- stage pre_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/mm/mm_attn_bm_out.txt \
--mlp_bp_methods_filein ${BMDIR}/mm/mm_mlp_bm_out.txt \
--patch_bp_methods_filein ${BMDIR}/mm/mm_patch_bm_out.txt \
--attn_bp_methods_fileout ${BMDIR}/ms/ms_attn_bm_out.txt \
--mlp_bp_methods_fileout ${BMDIR}/ms/ms_mlp_bm_out.txt \
--patch_bp_methods_fileout ${BMDIR}/ms/ms_patch_bm_out.txt

echo "MIDDLE SPARSE completed."


## final_sparse
mkdir -p ${BMDIR}/fs
touch ${BMDIR}/fs_attn_bm_out.txt
touch ${BMDIR}/fs_mlp_bm_out.txt
touch ${BMDIR}/fs_patch_bm_out.txt

CUDA_VISIBLE_DEVICES=3 python3 
-m torch.distributed.launch --nproc_per_node=1 \
--master_port=$RANDOM \--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 128 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/fs \
--pretrained_path ${BMDIR}/ms/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 1 -- stage pre_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/ms/ms_attn_bm_out.txt \
--mlp_bp_methods_filein ${BMDIR}/ms/ms_mlp_bm_out.txt \
--patch_bp_methods_filein ${BMDIR}/ms/ms_patch_bm_out.txt \
--attn_bp_methods_fileout ${BMDIR}/fs/fs_attn_bm_out.txt \
--mlp_bp_methods_fileout ${BMDIR}/fs/fs_mlp_bm_out.txt \
--patch_bp_methods_fileout ${BMDIR}/fs/fs_patch_bm_out.txt

echo "FINAL SPARSE completed."