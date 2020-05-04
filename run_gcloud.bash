IS_LOCAL=$1

JOBNAME=rl_job_hp_$(date -u +%y%m%d_%H%M%S)
REGION=europe-north1
BUCKET=robot_learning

if [ "$IS_LOCAL" == "local" ]
then
    gcloud ai-platform local train \
        --package-path=${PWD}/policy_gradient_robot_learning/learning \
        --module-name=policy_gradient_robot_learning.learning \
        --\
        --learning_rate=0.0001\
        --batch_size=13000\
        --episodes=3500\
        --output_dir='outputs'\
        --gpu=False
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
        --env="Pong-v4" \
        --model="NatureCNN" \
        --policy="PG" \
        --gpu=False \
        --episodes=3500 \
        --output_dir="gs://$BUCKET/policy_gradient_robot_learning"
fi

