import json
import pandas as pd
import cv2
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
from tqdm import tqdm
import torch
import logging
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from ppocr.utils.logging import get_logger
from model.salformer import SalFormer
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
import pandas as pd

ocr = PaddleOCR(use_angle_cls=True, lang="en") # need to run only once to download and load model into memory
logger = get_logger()
logger.setLevel(logging.ERROR)

READ_VALUE = 'read_value'
FILTERING = 'filter'
FIND_EXTREME = 'find_extreme'

ACTION = 20
IMAGE_SIZE = 320

sentenceTransformer = SentenceTransformer("stsb-roberta-base-v2")
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)
device = 'cpu' # cuda cpu
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
model = SalFormer(vit, bert).to(device)
checkpoint = torch.load('./VisSalFormer_weights.tar', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict(ques: str, chart_url: str):
    """
    Execute the prediction.

    Args:
        ques: a question string to feed into VisSalFormer

    Returns:
        heatmap from VisSalFormer (np.array)
    """
    image = Image.open(chart_url).convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer(ques, return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (mask * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (ACTION, ACTION))
    # heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    return heatmap

df = pd.read_csv("labels/taskvis_label.csv")

meta_data = []

# iterate each row of the df
for index, row in tqdm(df.iterrows()):
    print(row['id'], row['task'], row['chart'], row['question'])
    json_data = {
        "id": index+1,
        "image": row['chart'],
        "task": row['task'],
        "chart_type": "a horizontal bar chart",
        "question": row['question'],
        "text_positions": [],
    }

    chart_json = json.load(open('labels/%s.json'%(row['chart'])))
    task_json = json.load(open('labels/%s_%s.json'%(row['chart'], row['id'])))
    img_data = Image.open('labels/%s.png'%(row['chart']))
    save_path = 'images/chart_%s_origin.png'%(index+1)
    img_data.save(save_path)

    # axis, mark, title
    shapes = chart_json['shapes']
    grid_size = (img_data.size[1], img_data.size[0])

    title_map = np.zeros(grid_size)
    mark_map = np.zeros(grid_size)
    axis_map = np.zeros(grid_size)

    for shape in shapes:
        if shape['label'] == 'title':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            title_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
        if shape['label'] == 'mark':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            mark_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
        if shape['label'] == 'axis':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            axis_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
    title_map = cv2.resize(title_map, (ACTION, ACTION)) 
    cv2.imwrite('images/chart_%s_title.png'%(index+1), title_map)
    mark_map = cv2.resize(mark_map, (ACTION, ACTION)) 
    cv2.imwrite('images/chart_%s_mark.png'%(index+1), mark_map)
    axis_map = cv2.resize(axis_map, (ACTION, ACTION)) 
    cv2.imwrite('images/chart_%s_axis.png'%(index+1), axis_map)

    # search, map, map_mark, read, task_aoi
    shapes = task_json['shapes']
    search_map = np.zeros(grid_size)
    map_map = np.zeros(grid_size)
    map_mark_map = np.zeros(grid_size)
    read_map = np.zeros(grid_size)
    task_aoi_map = np.zeros(grid_size)

    for shape in shapes:
        if shape['label'] == 'search':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            search_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
            task_aoi_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
        if shape['label'] == 'map':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            map_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
            task_aoi_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
        if shape['label'] == 'map_mark':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            map_mark_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
        if shape['label'] == 'read':
            rect = [(int(shape['points'][0][0]), int(shape['points'][0][1])), (int(shape['points'][1][0]), int(shape['points'][1][1]))]
            read_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
            task_aoi_map[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255

    step_x = int(grid_size[0] / ACTION)
    step_y = int(grid_size[1] / ACTION)
    search_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    map_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    map_mark_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    read_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    task_aoi_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    for i in range(ACTION):
        for j in range(ACTION):
            if np.sum(search_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                search_action_map[i][j] = 255
                task_aoi_action_map[i][j] = 255
            if np.sum(map_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                map_action_map[i][j] = 255
                task_aoi_action_map[i][j] = 255
            if np.sum(map_mark_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                map_mark_action_map[i][j] = 255
            if np.sum(read_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                read_action_map[i][j] = 255
                task_aoi_action_map[i][j] = 255
    cv2.imwrite('images/chart_%s_search.png'%(index+1), search_action_map)
    cv2.imwrite('images/chart_%s_map.png'%(index+1), map_action_map)
    cv2.imwrite('images/chart_%s_map_mark.png'%(index+1), map_mark_action_map)
    cv2.imwrite('images/chart_%s_read.png'%(index+1), read_action_map)

    # saliency
    saliency_map = predict(row['question'], 'images/chart_%s_origin.png'%(index+1))
    cv2.imwrite('images/chart_%s_saliency.png'%(index+1), saliency_map)

    # peripheral
    origin_image = cv2.imread('images/chart_%s_origin.png'%(index+1))
    resized_chart = cv2.resize(origin_image, (IMAGE_SIZE, IMAGE_SIZE))
    cv2.imwrite('images/chart_%s.png'%(index+1), resized_chart)
    peripheral = cv2.resize(origin_image, (ACTION, ACTION))
    cv2.imwrite('images/chart_%s_p.png'%(index+1), peripheral)

    # semantic
    paddle_ocr_result = ocr.ocr(origin_image)
    semantic_map = np.zeros((grid_size[0], grid_size[1]), dtype=np.uint8)
    text_positions = []
    for line in paddle_ocr_result[0]:
        position = line[0]
        text = line[1][0]
        score = line[1][1]
        sentences = [
            row['question'],
            text,
        ]
        sentence_embeddings = sentenceTransformer.encode(sentences)
        similarity = cosine_similarity(sentence_embeddings[0].reshape(1, -1), sentence_embeddings[1].reshape(1, -1))
        semantic_similarity = float(similarity[0][0])
        if semantic_similarity < 0: semantic_similarity = 0
        color = (semantic_similarity*255) # , semantic_similarity*255, semantic_similarity*255
        cv2.rectangle(semantic_map, (int(position[0][0]), int(position[0][1])), (int(position[2][0]), int(position[2][1])), color, -1)
        position = [int(position[0][0] * IMAGE_SIZE / grid_size[1]), int(position[0][1] * IMAGE_SIZE / grid_size[0]), int(position[2][0] * IMAGE_SIZE / grid_size[1]), int(position[2][1] * IMAGE_SIZE / grid_size[0])]
        text_positions.append({"position": position, "text": text, "ocr_score": score, "semantic_similarity": semantic_similarity}) # , "score": score
    # cv2.imwrite('images/chart_%s_semantic_ocr.png'%id, ocr_image)
    semantic_map = cv2.resize(semantic_map, (ACTION, ACTION)) # 20 x 20
    semantic_map = semantic_map * axis_map # just care axis area
    max_similarity = np.max(semantic_map)
    if max_similarity > 0:
        semantic_map = semantic_map / max_similarity * 255

    cv2.imwrite('images/chart_%s_semantic.png'%(index+1), semantic_map)
    json_data['text_positions'] = text_positions
    meta_data.append(json_data)

    json.dump(json_data, open('images/chart_%s.json'%(index+1), 'w'))

with open('meta.json', 'w') as f:
    json.dump(meta_data, f)