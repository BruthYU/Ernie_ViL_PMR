
import jsonlines
from PIL import Image,ImageDraw,ImageFont
import os

data_path = "./data/PMR/test-ori-without-label.jsonl"

q_id = []
pred = []
label = []
cnt = 0
with open("./PMR/result_pmr_tv/13320.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        line_data = line.split('\t')
        q_id.append(int(line_data[0]))
        pred.append(int(line_data[1]))
        label.append(int(line_data[2]))
        if int(line_data[1])==int(line_data[2]):
            cnt += 1




data_item = []
data_img_id = []
data_label = []

for data_file in data_path:
    with open(data_file, "r+", encoding="utf8") as f:
        data_json = jsonlines.Reader(f)
        for item in data_json:
            data_item.append(item)
            data_img_id.append(item['img_id'])
            # data_label.append(item['answer_label'])

stat_ori = {'Distractor2':0, 'Distractor1':0, 'Action-False':0, 'Action-True':0}

pred_ori = pred
with jsonlines.open('./test_ori_without_label_result.jsonl','w') as f:
    for i,p in enumerate(pred_ori):
        line = {}
        item = data_item[i]
        line['total_id'] = item['total_id']
        line['image_id'] = data_img_id[i]
        line['prediction'] = pred_ori[i]
        print(line)
        f.write(line)

print("acc",cnt/len(q_id))
#{"total_id": 27, "img_id": "val-1159", "prediction": 0}



