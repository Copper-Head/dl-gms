# Must be run from the folder that contains it!
set -e

# Loads relevant variables from config file
source <(grep = config.ini)

NV_GPU=$GPUs nvidia-docker run -d \
          --name=$name \
          -l "lego-vae=true" \
          -v /srv/data/lgm2017-lego/orange_legos:$data_dir \
          -v $(pwd):/scripts \
          -v /srv/data/lgm2017-lego/weights:$weights_dir \
          -v /srv/data/lgm2017-lego/tensorboard-logdir:$tb_logdir \
          mlcog-up/tf-keras-trainer train.py

docker logs $name -f
