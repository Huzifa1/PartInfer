#!/bin/sh
DIR_PATH="$(cd .. && pwd)"

# Tunable Params
MODEL_NAME="llama3-3b"
METHOD="partinfer" # One of partinfer, coreinfer, dense, coreinfer_random_loading
USE_PARTINFER_IMPROVEMENTS=True

# PARTINFER Method Variables: 
LIMIT=10000000
TOKEN_SPARSITY=0.2
SPARSITY=0.4
START_NUM=4
END_NUM=26
BASE_NEURONS_PERCENT=0.3
BASE_NEURONS_TYPE="dataset"
LOADED_NEURONS_PERCENT=0.7
MODEL_NEURONS_FILEPATH="neuron_files/${MODEL_NAME}/model_neurons.json"
DATASET_NEURONS_FILEPATH="neuron_files/${MODEL_NAME}/qa.json"
MASK_FILEPATH="neuron_files/mask.pkl"


if [ "$METHOD" = "partinfer" ]; then
    USE_PARTINFER_IMPROVEMENTS=True
    FILE_PREFIX_NAME="partinfer"
elif [ "$METHOD" = "dense" ]; then
    if [ "$USE_PARTINFER_IMPROVEMENTS" = "True" ]; then
        FILE_PREFIX_NAME="dense"
    else
        FILE_PREFIX_NAME="reference"
    fi
elif [ "$METHOD" = "coreinfer" ]; then
    if [ "$USE_PARTINFER_IMPROVEMENTS" = "True" ]; then
        FILE_PREFIX_NAME="coreinfer_partial_loading"
    else
        FILE_PREFIX_NAME="coreinfer"
    fi
elif [ "$METHOD" = "coreinfer_random_loading" ]; then
    FILE_PREFIX_NAME="coreinfer_random_loading"
    METHOD="coreinfer"
    MODEL_NEURONS_FILEPATH="neuron_files/${MODEL_NAME}/random_neurons.json"
    BASE_NEURONS_PERCENT=$LOADED_NEURONS_PERCENT
    BASE_NEURONS_TYPE="model"
    USE_PARTINFER_IMPROVEMENTS=True
fi

run()
{
    echo "USE_PARTINFER_IMPROVEMENTS = $USE_PARTINFER_IMPROVEMENTS" > $DIR_PATH/transformers/partinfer_variables/partinfer_improvements.py
    cd $DIR_PATH/evaluation
    OUT_DIR="$DIR_PATH/results/${METHOD}_results/$TASK_NAME"
    mkdir -p $OUT_DIR
    if [ "$USE_PARTINFER_IMPROVEMENTS" = "True" ]; then
        python evaluate_task.py --model_name $MODEL_NAME --checkpoint_path $DIR_PATH/models/$MODEL_NAME/ --limit $LIMIT --task_name $TASK_NAME --output_path "$OUT_DIR/${FILE_NAME}" --token_sparsity $TOKEN_SPARSITY --sparsity $SPARSITY --method $METHOD --start_num $START_NUM --end_num $END_NUM --base_neurons_percent $BASE_NEURONS_PERCENT --base_neurons_type $BASE_NEURONS_TYPE --loaded_neurons_percent $LOADED_NEURONS_PERCENT --model_neurons_filepath $MODEL_NEURONS_FILEPATH --dataset_neurons_filepath $DATASET_NEURONS_FILEPATH --mask_filepath $MASK_FILEPATH
    else
        python evaluate_task.py --model_name $MODEL_NAME --checkpoint_path $DIR_PATH/models/$MODEL_NAME/ --limit $LIMIT --task_name $TASK_NAME --output_path "$OUT_DIR/${FILE_NAME}" --token_sparsity $TOKEN_SPARSITY --sparsity $SPARSITY --method $METHOD --start_num $START_NUM --end_num $END_NUM
    fi
    
    cd -
}


for TASK_NAME in "triviaqa" "squadv2" "piqa" "mlqa_en_en" "wmt16-de-en" "wmt16-ro-en" "cnn_dailymail" "xsum"; do

    if [ "$TASK_NAME" = "triviaqa" ] || [ "$TASK_NAME" = "squadv2" ] || [ "$TASK_NAME" = "piqa" ] || [ "$TASK_NAME" = "mlqa_en_en" ]; then
        DATASET_NEURONS_FILEPATH="neuron_files/$MODEL_NAME/qa.json"
    elif [ "$TASK_NAME" = "wmt16-de-en" ] || [ "$TASK_NAME" = "wmt16-ro-en" ]; then
        DATASET_NEURONS_FILEPATH="neuron_files/$MODEL_NAME/translate.json"
    elif [ "$TASK_NAME" = "xsum" ] || [ "$TASK_NAME" = "cnn_dailymail" ]; then
        DATASET_NEURONS_FILEPATH="neuron_files/$MODEL_NAME/summarize.json"
    fi

    FILE_NAME="${FILE_PREFIX_NAME}_${MODEL_NAME}.json"
    run
done

