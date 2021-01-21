source /users/visics/zwang/.bashrc
source /users/visics/zwang/miniconda3/etc/profile.d/conda.sh
conda activate base

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/local_gqa.py \
    --train train --valid "" \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --tqdm --output ./results/local_test \
    --load ./checkpoint/LXRT_official_baseline \
    --test testdev \
    --batchSize 1024 \
    --numWorkers 2 --mceLoss
