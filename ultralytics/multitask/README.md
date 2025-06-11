## python version 3.11.10

## dataset
- in 140.113.208.122 coachbox
    - training data
        - /hdd/dataset/alex_tracknet
        - /hdd/dataset/blion_tracknet
        - /hdd/dataset/profession_match_{n}
    - testing data
        - /hdd/dataset/profession_match_{n}_test

### Folder layout

Each match directory under the dataset root should include at least:

```
match_x/
├── annotation.json        # frame annotations
├── frame/                 # extracted PNG frames
│   ├── 000001.png
│   └── ...
├── csv/                   # ball trajectory CSV files (optional)
└── video/                 # source videos (optional)
```

The number of frames loaded from each `match_x` folder can be limited by
adjusting the optional `path_counts` dictionary in
`MultiTaskConfigurableDataset`.

## how to build
ULTRALYTICS_BRANCH: 指定使用分之
CACHE_BUSTER: 確保每次都能拉取最新的分支
```
docker build \
    -f ultralytics/tracknet/Dockerfile \
    --build-arg ULTRALYTICS_BRANCH=feat/p3p4p5-test2 \
    --build-arg CACHE_BUSTER=$(date +%s) \
    -t tracknet1000:latest .
```

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

python multitask.py --mode train --model_path ultralytics/models/v8/multitask.yaml --data ultralytics/datasets/multitask.yaml --epochs 200

```

The dataset root is defined by the `path` field in `multitask.yaml`. You can
edit the file directly or supply a custom YAML on the command line via the
`--data` argument:

```
python multitask.py --data path/to/your_multitask.yaml
```

## TODO
- 挑影片的方式目前是寫死的
