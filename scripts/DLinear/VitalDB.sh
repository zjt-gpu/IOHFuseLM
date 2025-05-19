seq_len=300
model=DLinear

python main.py \
    --root_path ./dataset/VitalDB/Sampling_3s \
    --desription_path ./dataset/VitalDB \
    --data_path all_patient.csv \
    --split_name cutted3sinput15mpredict5m \
    --dataset_name VitalDB \
    --model_path ./GPT2_VitalDB \
    --data Blood \
    --seq_len $seq_len \
    --pred_len 100 \
    --batch_size 8 \
    --finetune_learning_rate 0.0001 \
    --decay_fac 0.75 \
    --dropout 0.2 \
    --itr 3 \
    --model $model \
    --tmax 10 \
    --if_augmentation 0 \
    --if_pretrain 0 \
    --device_id cuda:1

python main.py \
    --root_path ./dataset/VitalDB/Sampling_3s \
    --desription_path ./dataset/VitalDB \
    --data_path all_patient.csv \
    --split_name cutted3sinput15mpredict10m \
    --dataset_name VitalDB \
    --model_path ./GPT2_VitalDB \
    --data Blood \
    --seq_len $seq_len \
    --pred_len 200 \
    --batch_size 8 \
    --finetune_learning_rate 0.0001 \
    --decay_fac 0.75 \
    --dropout 0.2 \
    --itr 3 \
    --model $model \
    --tmax 10 \
    --if_augmentation 0 \
    --if_pretrain 0 \
    --device_id cuda:1

python main.py \
    --root_path ./dataset/VitalDB/Sampling_3s \
    --desription_path ./dataset/VitalDB \
    --data_path all_patient.csv \
    --split_name cutted3sinput15mpredict15m \
    --dataset_name VitalDB \
    --model_path ./GPT2_VitalDB \
    --data Blood \
    --seq_len $seq_len \
    --pred_len 300 \
    --batch_size 8 \
    --finetune_learning_rate 0.0001 \
    --decay_fac 0.75 \
    --dropout 0.2 \
    --itr 3 \
    --model $model \
    --tmax 10 \
    --if_augmentation 0 \
    --if_pretrain 0 \
    --device_id cuda:1