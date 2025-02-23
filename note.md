yolov8
    ultralytics\nn\modules\head.py
        調整 detect 輸出，減少長寬以及增加x,y變量

    loss function 修改

    ultralytics\yolo\cfg 修改

    training 的 output weight 要注意 要加重

可以從 dataloader 知道 10 張圖片怎麼丟進去
train_data
    match1
        video
            xxx.mp4
        frame
            frame<ID>.png
        ball_trajectory
            csv file: frame,x,y
    match2
        video
            xxx.mp4
        frame
            frame<ID>.png
        ball_trajectory
            csv file: frame,x,y


程式碼解析
    nn.module parse_model
        解析 models.yaml 轉換成一層一層的網路，輸入的 channel 可以用參數 ch 帶入
    yaml 的最後一層，會接到 Detect 模型，目前需要修改這邊，讓他可以輸出我要的 10 組(x,y,dx,dy,conf)
    data 的部分怎麼塞進來，也是一大問題

    loss
        FL(pi​,ti​)=−αti​​(1−pi​)γlog(pi​) 球存在
        FL(pi​,ti​)=−αti​​piγ​log(1−pi​)   球不存在

## idea
- 每個網格的x,y 大小會有限制 (需要Sigmoid)
- from yoloX: 可以把 cell 中心四周的 cell (3*3)也當作 positive 去算 loss
- 是否可以使用 upsampling 做一些事情
- Kalman Filtering


## train record
- train4: 10/9 使用 non sigmoid predict run 50 epoch
- train1: 嘗試使用 sigmoid(x, y, dx, dy) 只跑了個數的 epoch
- train6: 使用 weight=100
- train49: weight=100, 16 batch, fix focal loss, epoch 50,commit ba2d8ff
- train71: 11/16 tanh, bce loss, epoch 50 commit bd47c9b
- train79: 圖像化，50 epoch，可用來測試 conf loss
- train81: 圖像化，100 epoch，可用來測試 conf loss
- train83: 圖像化，100 epoch，可用來測試 conf loss，包含其他數字
- C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img5: conf 修正完成
- train148 加入所有 loss 100 epoch
- train181 pos+conf loss 50 epoch (head不拆分) check_training_img9
- train183 pos+conf+mov loss 47 epoch (head不拆分) check_training_img10
- train196 pos+conf+mov loss 100 epoch (head不拆分) check_training_img11, commit: f1aecf0 (lr=0.01, momentum=0.9)
- train226 整理後架構 100 epoch

feats = 三個 tensor[16*144*80*80],[16*144*40*40],[16*144*20*20] 
no=144
reg_max=16
nc=80

## 20240518
- 在 predict 的時候，dataset 不知道為什麼會變成 len(0)
- 確認 predict 有問題，有可能是模型沒有儲存到，因為我拿 training data 去 predict，但是結果與 training 的時候不同，目前想法有以下:
    - 拿 best.pt 繼續 train，看看結果長什麼樣子，如果異常，代表 model 沒有正常儲存
    - 檢查 predict 程式碼哪裡異常

## 20240519
- 拿模型 pt 進行訓練，看看訓練時的數據，是否符合預期，以確認 pt 是否有正確儲存

## 20240830
- dxdy 有加與沒加入，對於模型學習的能力是否有正向影響

## 20240830
- train238 測試 epoch 200 with dxdy
- coachbox 密碼: nol56680

## docker debug
### 空間不足
```
docker container prune

//查看 /var/lib/docker/ 目錄及其子目錄的大小
sudo du -sh /var/lib/docker/

//執行以下命令來查看包含 /var/lib/docker/ 的分區的總空間及可用空間
df -h /var/lib/docker/
docker exec -it 0c3e1511776e /bin/bash

```

## 20240907
- train253 測試 epoch 200 without dxdy
- Huber Loss 或 Smooth L1 Loss

## 20240917
- train263 epoch 200 使用 DFL without dxdy and hit loss

## 20240919
待辦
- 確認 263 有多少個 > 0.5 的 conf
- 加入 IOU loss (Gaussian IOU)

## 20240920
- train264 epoch 200 iou sigma=0.7

## 20240923
- train266 epoch 200, 移除 IOU, penalty 調整 TH = 0.5

## 20240925
- train267 epoch 200, FP penalty = 4000

## 20240929
- train278 epoch 1xx, 改 conf loss 但是太慢 (結果是 Dataset 拿錯)
- train286 epoch 6x, focal loss
- train288 epoch 59, focal loss with FN FP 權重
- train294 epoch 200, focal loss with FN FP 權重

## 20241002
- train302 epoch 200, focal loss with 最外層的 weight fb113b3

## 20241005
- train318 epoch 200, DFL + conf + dxdy: 0382e0d

## 20241006
- train322 epoch 200, DFL + conf + dxdy: eba7aca
- train328 epoch 200, DFL + conf

## 20241021
- train341 epoch 200, DFL + conf: 4eec5a6, val mode + save val (此 val 是 training 沒有看過的資料)

## 20241027
- val total ball count = 3904

## 20241027
- train348 epoch 200, conf+xy val 目前最佳, 可拿來做為之後比較依據 (ac1ec91fd13b772af3670146b9c33fbdb674d7d0)


## 20241101
- train354 epoch 200, conf+xy, 增加快速球的權重 dist = 20

12:50 包含 之前

## 20241104
- train371 60FPS + 120 FPS


## 20241110
- 距離相差2倍 hitV2- TP: 143, FP: 65, TN: 1806, FN: 236
- 距離相差1.5倍 

## 20241118
- train429 f205932a7638d524ab513846956c37d334264d20
  - 修好 fitness
  - 找到 loss 有負數的原因
  - precision-recall 圖，正確產出
- train431 35583f087ad425f97d86c025971839c6efd064a5
  - 修好 FP 的計算，IOU 太小也算預測有球但預測錯誤

## 20241123
- Ctrl + P 再加上 Ctrl + Q
  - 可以不中斷 python 執行
  - docker exec <container_name> ps aux => 可以檢查容器內執行的進程

- 執行 python script.py & => Ctrl + P 再加上 Ctrl + Q
  - 可以讓 python 在背景執行
- train437 b91725148efcca01f362cf3a55e7a975f7045c92
  - 不使用 conf weight
  - hit duplicate: 5

## 20241129
- train440 weight+hit duplicate: 5
  - 效果沒有比較好

## 20241201
*更換分支到 feat/detect-on-p3*
- train442
  - use p3
  - only weight (without hit duplicate)
- 單一 cell 只能偵測一個相同的物品
- w h 可以大於 cell 大小
  - .matmul(self.proj.type(pred_dist.dtype))
  - 這裡的 self.proj，決定的物品最大範圍 （離散的區間大小）

- train443 120, 60, 40 FPS (其餘同 442 程式碼)
  - f22ecbe6597421e23eab493e513caac84b4bb6e0

- blion: 1_05_03 271, 272 兩個 frame 相同

## 20241203
- TODO
- 修正 val loss

# 20241208
- train450
  - 改為中心點 anchor

- train451
  - add next x, y prediction
  - in the same head with current x, y
- train473
  - 修復 next xy
  - print next (x, y) when validate
- train475
  - adjust conf weight, more focus on FN
  - change focal loss hyper param (0.85, 1.5) 

# 20250120
- train502
  - 綜合可選的多樣 dataset 資料進行訓練
  - 尚未實作隨機旋轉與縮放
  - 關閉 nms (效能問題，待解決)

# 20250219
- training 點子
  - 將同一份 dataset 切成 n 份，拿第 i 份訓練結果，當作第 i+1 份訓練時的 penalty 權重依據

# 20250221
- train506 -- epoch 200 (9e2c28f6efca44aeedf098231c752068d1e2827e)
  - 引用 tracknetv3 Background Estimation
  - ```
    docker run --gpus all --ipc=host \
    -v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
    -v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/profession_match_2 \
    -v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/profession_match_3 \
    -v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/profession_match_4 \
    -v /hdd/dataset/ces2025_all_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -it tracknetv4
    ```

-train517 (650c5fb7262aa2fa8a00c4303179d5364c18e20d)
  - fix 影像疊重疊問題
  - cache 不使用 docker 空間，使用 docker volume
  - fix val image tensor 沒有正規劃的情形 （應該是導致 val loss 異常的主因）
  - ```
    docker run --gpus all --ipc=host \
    -v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
    -v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/profession_match_2 \
    -v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/profession_match_3 \
    -v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/profession_match_4 \
    -v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_5 \
    -v /hdd/dataset/profession_match_2:/usr/src/datasets/tracknet/train_data/profession_match_6 \
    -v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_7 \
    -v /hdd/dataset/profession_match_4:/usr/src/datasets/tracknet/train_data/profession_match_8 \
    -v /hdd/dataset/profession_match_5:/usr/src/datasets/tracknet/train_data/profession_match_9 \
    -v /hdd/dataset/profession_match_6:/usr/src/datasets/tracknet/train_data/profession_match_10 \
    -v /hdd/dataset/profession_match_7:/usr/src/datasets/tracknet/train_data/profession_match_11 \
    -v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/train_data/profession_match_12 \
    -v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.cache:/usr/src/datasets/tracknet/train_data/.cache \
    -it tracknetv4
    ```