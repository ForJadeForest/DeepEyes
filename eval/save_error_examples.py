import json
import os
import re
from pprint import pprint

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

type_name = "relative_position"

result_list = []
with open(
    f"infer_result/deepeyes_fix_2/result_{type_name}_deepeyes_fix_2_acc.jsonl", "r"
) as f:
    for line in f:
        data = json.loads(line)
        result_list.append(data)

img_root = f"data/vstar_bench/{type_name}"

save_dir = f"./bad_case_analysis/vstar/{type_name}"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "bad_case"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "good_case"), exist_ok=True)


font = ImageFont.truetype("DejaVuSans.ttf", 20)

def convert_bbox_format(bbox):
    """Convert bbox to [x0, y0, x1, y1] format where x1 > x0 and y1 > y0"""
    if len(bbox) != 4:
        return None

    # 如果是[x, y, w, h]格式，转换为[x0, y0, x1, y1]
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        x0, y0 = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]
        return [x0, y0, x0 + w, y0 + h]

    # 确保x1 > x0且y1 > y0
    x0, y0, x1, y1 = bbox
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return [x0, y0, x1, y1]


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 可视化模型输出结果

# 原图 + 模型预测BBox + GT BBox
# Question + options + GT answer
# Model Output

# 获取原始问题
for i in range(len(result_list)):

    item = result_list[i]
    img_path = os.path.join(img_root, item["image"])
    idx = result_list[i]["image"].split(".")[0]
    question = result_list[i]["pred_output"][1]["content"][1]["text"].split(
        "\n\n\nThink"
    )[0]
    GT_answer = result_list[i]["answer"]

    model_output = result_list[i]["pred_output"][2:]

    model_output_text = tokenizer.apply_chat_template(
        model_output, tokenize=False
    ).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")

    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    match = re.search(pattern, model_output_text)
    # 找到所有tool_call
    tool_call_list = re.findall(pattern, model_output_text)

    bbox_list = []
    for tool_call in tool_call_list:
        bbox_list.append(json.loads(tool_call)["arguments"])

    # 在原始图像画出 bbox
    for bbox in bbox_list:
        bbox_points = bbox["bbox_2d"]
        lebel = bbox["label"]
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error: Failed to open image file: {img_path}, error: {str(e)}")
            continue
        anno_path = os.path.join(img_root, item["image"].replace(".jpg", ".json"))
        gt_bbox_list = []
        gt_label_list = []
        
        if os.path.exists(anno_path):
            try:
                with open(anno_path, "r") as f:
                    anno = json.load(f)
                    if "bbox" in anno and "target_object" in anno:
                        gt_bbox_list = anno["bbox"]
                        gt_label_list = anno["target_object"]
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse annotation file: {anno_path}")
            except Exception as e:
                print(f"Warning: Error reading annotation file: {anno_path}, error: {str(e)}")
        else:
            print(f"Warning: Annotation file not found: {anno_path}")

        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_points, outline="red", width=4)
        draw.text((bbox_points[0], bbox_points[1]), lebel, fill="red", font=font)
        
        for gt_bbox, gt_label in zip(gt_bbox_list, gt_label_list):
            converted_bbox = convert_bbox_format(gt_bbox)
            if converted_bbox is not None:
                draw.rectangle(converted_bbox, outline="green", width=4)
                draw.text((converted_bbox[0], converted_bbox[1]), gt_label, fill="green", font=font)
            else:
                print(f"Warning: Invalid bbox format: {gt_bbox}")

    # save question
    item_info = {
        "question": question,
        "GT_answer": GT_answer,
        "model_output": model_output_text,
        "bbox_list": bbox_list,
        "gt_bbox_list": gt_bbox_list,
        "gt_label_list": gt_label_list,
    }

    # save GT answer
    if item["acc"] == 0:
        save_path = os.path.join(save_dir, "bad_case", idx)
    else:
        save_path = os.path.join(save_dir, "good_case", idx)
    os.makedirs(save_path, exist_ok=True)
    img.save(f"{save_path}/bbox_img.png")
    with open(f"{save_path}/item_info.txt", "w") as f:
        f.write("Question: " + question + "\n\n")
        f.write("GT Answer: " + GT_answer + "\n\n")
        f.write("Model Output: " + model_output_text + "\n\n")
        f.write("BBox List: " + str(bbox_list) + "\n\n")
        f.write("GT BBox List: " + str(gt_bbox_list) + "\n\n")
        f.write("GT Label List: " + str(gt_label_list) + "\n\n")
