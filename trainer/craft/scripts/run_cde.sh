# sed -i -e 's/\r$//' scripts/run_cde.sh
EXP_NAME=custom_data_release_test_3
yaml_path="config/$EXP_NAME.yaml"
cp config/custom_data_train.yaml $yaml_path
#CUDA_VISIBLE_DEVICES=0,1 python3 train_distributed.py --yaml=$EXP_NAME --port=2468
CUDA_VISIBLE_DEVICES=0 python3 train.py --yaml=$EXP_NAME --port=2468
rm "config/$EXP_NAME.yaml"