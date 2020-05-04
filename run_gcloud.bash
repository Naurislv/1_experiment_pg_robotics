
# Input arguments
IS_LOCAL=$1

# Job variables
JOBNAME=rl_job_hp_$(date -u +%y%m%d_%H%M%S)
REGION=europe-north1
BUCKET=robot_learning

# Training arguments
SEED=111
EPISODES=3500
GPU=False

# Training arguments (only local)
LEARNING_RATE=0.0001
BATCH_SIZE=13000

# Training arguments (only global)
OUTPUT_DIR="gs://$BUCKET/policy_gradient_robot_learning"


if [ "$IS_LOCAL" == "local" ]
then
    gcloud ai-platform local train \
        --package-path=${PWD}/policy_gradient_robot_learning/learning \
        --module-name=learning \
        --\
        --learning_rate=$LEARNING_RATE \
        --batch_size=$BATCH_SIZE \
        --episodes=$EPISODES \
        --gpu=$GPU \
        --seed=$SEED
else
    gcloud ai-platform jobs submit training $JOBNAME \
        --package-path=${PWD}/policy_gradient_robot_learning \
        --module-name=policy_gradient_robot_learning.learning \
        --region=$REGION \
        --staging-bucket=gs://$BUCKET \
        --config=hyperparam.yaml \
        --runtime-version 2.1 \
        --python-version 3.7 \
        --\
        --gpu=$GPU \
        --episodes=$EPISODES \
        --output_dir=$OUTPUT_DIR \
        --seed=$SEED
fi

