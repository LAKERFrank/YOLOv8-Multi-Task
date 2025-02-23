## python version 3.11.10

## dataset
- in 140.113.208.122 coachbox
- /hdd/dataset/alex_tracknet
- /hdd/dataset/blion_tracknet

## how to run
```
docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
-v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
-it tracknetv4

python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 200

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train345/weights/last.pt --source /usr/src/datasets/tracknet/val_data
```

## TODO
- 挑影片的方式目前是寫死的