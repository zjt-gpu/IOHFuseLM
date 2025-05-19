seq_len=300
model=IOHFuseLM

python main.py \
    --root_path ./dataset/VitalDB/Sampling_3s \
    --desription_path ./dataset/VitalDB \
    --data_path all_patient.csv \
    --split_name cutted3sinput15mpredict5m \
    --dataset_name VitalDB \
    --data Blood \
    --seq_len $seq_len \
    --pred_len 100 \
    --batch_size 8 \
    --pretrain_learning_rate 0.00003 \
    --finetune_learning_rate 0.0001 \
    --gpt_layers 3 \
    --aug_nums 2 \
    --decay_fac 0.75 \
    --dropout 0.2 \
    --itr 3 \
    --model $model \
    --tmax 10 \
    --sampling_rate 3 \
    --model_path ./GPT2_VitalDB \
    --max_txt_len 80 \
    --device_id cuda:7

python main.py \
    --root_path ./dataset/VitalDB/Sampling_3s \
    --desription_path ./dataset/VitalDB \
    --data_path all_patient.csv \
    --split_name cutted3sinput15mpredict10m \
    --dataset_name VitalDB \
    --data Blood \
    --seq_len $seq_len \
    --pred_len 200 \
    --batch_size 8 \
    --pretrain_learning_rate 0.00005 \
    --finetune_learning_rate 0.0001 \
    --gpt_layers 4 \
    --aug_nums 2 \
    --decay_fac 0.75 \
    --dropout 0.2 \
    --itr 3 \
    --model $model \
    --tmax 10 \
    --sampling_rate 3 \
    --model_path ./GPT2_VitalDB \
    --max_txt_len 80 \
    --device_id cuda:7

python main.py \
    --root_path ./dataset/VitalDB/Sampling_3s \
    --desription_path ./dataset/VitalDB \
    --data_path all_patient.csv \
    --split_name cutted3sinput15mpredict15m \
    --dataset_name VitalDB \
    --data Blood \
    --seq_len $seq_len \
    --pred_len 300 \
    --batch_size 8 \
    --pretrain_learning_rate 0.00001 \
    --finetune_learning_rate 0.0001 \
    --gpt_layers 5 \
    --aug_nums 2 \
    --decay_fac 0.75 \
    --dropout 0.2 \
    --itr 3 \
    --model $model \
    --tmax 10 \
    --sampling_rate 3 \
    --model_path ./GPT2_VitalDB \
    --max_txt_len 80 \
    --device_id cuda:7