import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import re


class InferenceModel:
    def __init__(self, model_path='/root/models/agent_result_vis'):
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def inference(self, image, text):
        pixel_values = self.load_image(image, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        question = text
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response    

def visualize(image, result):
    img = Image.open(image)
    w, h = img.size
    action_name, xy_touch, xy_lift, type_text = result
    plt.imshow(img)
    plt.title(f'{action_name}')
    if xy_touch is not None:
        touch = ((xy_touch[0]+xy_touch[2])/2000*w, (xy_touch[1]+xy_touch[3])/2000*h)
        plt.plot([touch[0]], [touch[1]], 'ro')
    if xy_lift is not None:
        lift = ((xy_lift[0]+xy_lift[2])/2000*w, (xy_lift[1]+xy_lift[3])/2000*h)
        plt.plot([touch[0], lift[0]], [touch[1], lift[1]], '--b')
    if type_text is not None:
        plt.title(f'Type: {type_text}')
    plt.axis('off')
    plt.show()

def parse(text):
    # Return (action_name, xy_touch, xy_lift, type_text)
    action = None
    xy_touch = None
    xy_lift = None
    type_text = None
    bbox_p = re.compile('\[\[([^]]+)\]\]')
    try:
        if len(text) >= 4 and text[:4] == 'type':
            action = 'type'
            type_text = text[4:].strip()
        elif len(text) >= 5 and text[:5] == 'click':
            action = 'click'
            xy_touch = bbox_p.findall(text)[-1]
            xy_touch = xy_touch.replace(' ', '').split(',')
            xy_touch = [int(d) for d in xy_touch]
        elif len(text) >= 10 and text[:10] == 'dual_point':
            action = 'dual_point'
            xy_touch, xy_lift = bbox_p.findall(text)[-2:]
            xy_touch = xy_touch.replace(' ', '').split(',')
            xy_lift = xy_lift.replace(' ', '').split(',')
            xy_touch = [int(d) for d in xy_touch]
            xy_lift = [int(d) for d in xy_lift]
        elif len(text) >= 10 and text[:10] == 'press_back':
            action = 'press_back'
        elif len(text) >= 10 and text[:10] == 'press_home':
            action = 'press_home'
        elif len(text) >= 11 and text[:11] == 'press_enter':
            action = 'press_enter'
        elif len(text) >= 13 and text[:13] == 'task_complete':
            action = 'task_complete'
        elif len(text) >= 15 and text[:15] == 'task_impossible':
            action = 'task_impossible'
        return (action, xy_touch, xy_lift, type_text)
    except:
        return (None, None, None, None)

import gradio as gr

def predict_next_action(img, instruction):
    return 

from flask import Flask, request, jsonify
from PIL import Image

inference = InferenceModel()

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    if 'text' not in request.form or 'image' not in request.files:
        return jsonify({"error": "No text or image provided"}), 400
    
    # Get the text from the form data
    text_data = request.form['text']
    
    # Get the image from the files
    image_file = request.files['image']
    #image = Image.open(image_file.stream)
    
    # Process image (for example, get its size)
    #image_size = image.size  # you can do more complex processing here
    
    # Process text (you can use it in your model or logic)
    response = inference.inference(image_file, text_data)
    processed_text = f"{response}"
    
    # Respond with the processed text
    return jsonify({"response": processed_text})

if __name__ == '__main__':
    # Run the Flask app on port 5678
    app.run(debug=True, port=5678)

