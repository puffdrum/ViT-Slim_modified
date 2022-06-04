#!/bin/bash

## early remember early forgetting
## nohup sh s-eref_test.sh ./sparse_models/s-eref-bs64_<config> > logs/s-eref-bs64_<config>.log 2>&1 &
## <config>: ps_epochs - middle_times - mm_epochs - ms_epochs - fs_epochs
## e.g. nohup sh sparse_s-eref_bs64_10-3-10-10-20.sh ./sparse_models/s-eref-bs64_10-3-10-10-20 > logs/s-eref-bs64_10-3-10-10-20.log 2>&1 &

BMDIR=${1} # OUTPUT DIR

mkdir -p ${BMDIR}

## pre_sparse
mkdir -p ${BMDIR}/ps

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/ps \
--pretrained_path pretrained_models/deit_small_patch16_224-cd65a155.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage pre_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_fileout ${BMDIR}/ps/ps_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/ps/ps_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/ps/ps_patch_bm_out

echo "PRE SPARSE completed."

sleep 10

## middle_mix 1
mkdir -p ${BMDIR}/mm1

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/mm1 \
--pretrained_path ${BMDIR}/ps/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage middle_mix --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/ps/ps_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/ps/ps_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/ps/ps_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/mm1/mm1_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/mm1/mm1_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/mm1/mm1_patch_bm_out

echo "MIDDLE MIX 1 completed."

sleep 10

## middle_sparse 1
mkdir -p ${BMDIR}/ms1

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/ms1 \
--pretrained_path ${BMDIR}/mm1/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage middle_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/mm1/mm1_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/mm1/mm1_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/mm1/mm1_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/ms1/ms1_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/ms1/ms1_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/ms1/ms1_patch_bm_out

echo "MIDDLE SPARSE 1 completed."

sleep 10

## middle_mix 2
mkdir -p ${BMDIR}/mm2

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/mm2 \
--pretrained_path ${BMDIR}/ms1/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage middle_mix --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/ms1/ms1_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/ms1/ms1_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/ms1/ms1_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/mm2/mm2_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/mm2/mm2_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/mm2/mm2_patch_bm_out

echo "MIDDLE MIX 2 completed."

sleep 10

## middle_sparse 2
mkdir -p ${BMDIR}/ms2

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=$15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/ms2 \
--pretrained_path ${BMDIR}/mm2/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage middle_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/mm2/mm2_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/mm2/mm2_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/mm2/mm2_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/ms2/ms2_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/ms2/ms2_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/ms2/ms2_patch_bm_out

echo "MIDDLE SPARSE 2 completed."

sleep 10

## middle_mix 3
mkdir -p ${BMDIR}/mm3

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/mm3 \
--pretrained_path ${BMDIR}/ms2/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage middle_mix --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/ms2/ms2_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/ms2/ms2_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/ms2/ms2_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/mm3/mm3_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/mm3/mm3_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/mm3/mm3_patch_bm_out

echo "MIDDLE MIX 3 completed."

sleep 10

## middle_sparse 3
mkdir -p ${BMDIR}/ms3

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/ms3 \
--pretrained_path ${BMDIR}/mm3/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 10 --stage middle_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/mm3/mm3_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/mm3/mm3_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/mm3/mm3_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/ms3/ms3_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/ms3/ms3_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/ms3/ms3_patch_bm_out

echo "MIDDLE SPARSE 3 completed."

sleep 10

## final_sparse
mkdir -p ${BMDIR}/fs

CUDA_VISIBLE_DEVICES=0,3 python3 \
-m torch.distributed.launch --nproc_per_node=2 \
--master_port=15000 \
--use_env search_eref.py \
--model deit_small_patch16_224 --batch-size 64 \
--data-set IMNET --data-path ../Dataset/Imagenet1k/ \
--output_dir ${BMDIR}/fs \
--pretrained_path ${BMDIR}/ms3/running_ckpt.pth \
--w1 2e-4 --w2 5e-5 --w3 1e-4 --epochs 20 --stage final_sparse --ft 0.1 --rt 0.5 \
--attn_bp_methods_filein ${BMDIR}/ms3/ms3_attn_bm_out \
--mlp_bp_methods_filein ${BMDIR}/ms3/ms3_mlp_bm_out \
--patch_bp_methods_filein ${BMDIR}/ms3/ms3_patch_bm_out \
--attn_bp_methods_fileout ${BMDIR}/fs/fs_attn_bm_out \
--mlp_bp_methods_fileout ${BMDIR}/fs/fs_mlp_bm_out \
--patch_bp_methods_fileout ${BMDIR}/fs/fs_patch_bm_out

echo "FINAL SPARSE completed."