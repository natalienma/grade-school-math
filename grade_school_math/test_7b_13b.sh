export MASTER_ADDR=localhost
export MASTER_PORT=2131
export CUDA_VISIBLE_DEVICES=0,1
MODEL_DIR=$1
OUT_DIR=$1

python3 -m torch.distributed.run --nproc_per_node 2 --master_port 7834 test.py \
                        --base_model $MODEL_DIR \
                        --data_path "./data/test_use.jsonl" \
                        --out_path $OUT_DIR \
                        --batch_size 64