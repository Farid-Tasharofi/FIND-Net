DATA_PATH="/home/woody/iwi5/iwi5200h/dataset/26.06/"
BATCH_SIZE=1
BATCHNUM=700
PATCH_SIZE=512
FOURIER_INPUT_SIZE=512
NUM_M=32
NUM_Q=32
T=3
S=10
ETAM=1
ETAX=5
USE_GPU=true
GPU_ID="0"
LOG_DIR="./logs/"
CHECKPOINT="checkpoint.pt"
MANUAL_SEED=""  # Optional: If defined, append in Python command
MASKEDEVAL=true # This will calculate metrics only on the non-metal regions
MODEL_DIRECTORY="./pretrained_models/FINDNet_no_GF/"
TEST_INFO="/test_FINDNet/"  # for fine tuning --> "/06.09_MX-net_FFC_128_fine/"  /  for scratch training --> ""

# Execute Python script with all parameters
python test_FINDNet.py \
  --data_path "$DATA_PATH" \
  --batchSize $BATCH_SIZE \
  --patchSize $PATCH_SIZE \
  --batchnum $BATCHNUM \
  --num_M $NUM_M \
  --num_Q $NUM_Q \
  --T $T \
  --S $S \
  --etaM $ETAM \
  --etaX $ETAX \
  --use_gpu $USE_GPU \
  --gpu_id "$GPU_ID" \
  --log_dir "$LOG_DIR" \
  $(if [ ! -z "$MANUAL_SEED" ]; then echo "--manualSeed $MANUAL_SEED"; fi) \
  $(if $MASKEDEVAL; then echo "--masked_eval"; fi) \
  --model_dir "$MODEL_DIRECTORY" \
  --checkpoint "$CHECKPOINT" \
  --test_info "$TEST_INFO"