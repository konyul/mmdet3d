
#python3 tools/train.py configs/Faster_config/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py --work-dir Fast/

#CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/dist_train.sh configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py 4 --work-dir logs/
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/Faster_config/faster_rcnn_r101_fpn_1x_coco.py 4 --work-dir faster_rcnn_r101_fpn_1x_coco/
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/Faster_config/faster_rcnn_x101_32x4d_fpn_1x_coco.py 4 --work-dir faster_rcnn_x101_32x4d_fpn_1x_coco/
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/Faster_config/faster_rcnn_x101_64x4d_fpn_1x_coco.py 4 --work-dir faster_rcnn_x101_64x4d_fpn_1x_coco

#CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/dist_test.sh configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py logs0.0002/epoch_9.pth 4 --eval mAP


#CUDA_VISIBLE_DEVICES=0,3 GPUS=2 ./tools/dist_test.sh configs/retinanet/retinanet_r50_fpn_1x_jrdb.py work_dirs/retinanet_r50_fpn_1x_jrdb/epoch_2.pth 2 --format-only --options pklfile_prefix=./results/result.pkl submission_prefix=./data/label_2



CUDA_VISIBLE_DEVICES=1 python3 tools/test.py configs/retinanet/retinanet_r50_fpn_1x_jrdb.py work_dirs/retinanet_r50_fpn_1x_jrdb/epoch_2.pth --format-only --options pklfile_prefix=./results/result.pkl submission_prefix=./data/label_2
#CUDA_VISIBLE_DEVICES=3 python3 tools/test.py configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py logs0.0002/epoch_9.pth --format-only --options pklfile_prefix=./results/result.pkl submission_prefix=./data/label_2_faster
#CUDA_VISIBLE_DEVICES=3 python3 tools/test.py configs/DH_config/dh_faster_rcnn_r50_fpn_1x_coco.py data/Double_head0.0002/epoch_3.pth --format-only --options pklfile_prefix=./results/result.pkl submission_prefix=./data/label_2_faster
#CUDA_VISIBLE_DEVICES=0,3 GPUS=2 ./tools/dist_train.sh configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_kitti.py 2
#./tools/dist_test.sh configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py     logs0.0002/epoch_9.pth    4  --format-only --options "pklfile_prefix=./results/result.pkl submission_prefix=./data/label_2_0.5"

# python3 ./data/convert.py
# python3 ./data/draw.py

#python3 ./data/JRDB/results/sibal.py


#CUDA_VISIBLE_DEVICES=2 python3 tools/test.py configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py logs0.0002/epoch_9.pth --show-dir ./demo_0.6 --show-score-thr 0.6

#./tools/dist_test.sh configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py logs0.0002/epoch_9.pth 4 --eval mAP

#CUDA_VISIBLE_DEVICES=1 python3 tools/test.py configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py logs0.0002/epoch_9.pth --show-dir ./realdemo_0.4 --show-score-thr 0.4

# ./tools/dist_test.sh configs/DH_config/dh_faster_rcnn_r50_fpn_1x_coco.py ./data/Double_head0.0001/epoch_3.pth 4 --eval mAP
# ./tools/dist_test.sh configs/DH_config/dh_faster_rcnn_r50_fpn_1x_coco.py ./data/Double_head0.0002/epoch_3.pth 4 --eval mAP

python3 alarm.py