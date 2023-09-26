docker run --gpus 2 -p 6667:6667 -v $(pwd):/app/my_picard --name my_picard_dev -it 4b699d75f63a

python ./seq2seq/run_seq2seq.py ./configs/train.json

docker build --build-arg BASE_IMAGE=pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel --tag local_picard .