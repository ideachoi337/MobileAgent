# MobileAgent
🤗 [HF_link: Download model](https://huggingface.co/ideachoi/InternVL_MobileAgent) 

Multimodal Large-Language-Model for Mobile Agent.
The model is fine-tuned from Intern2-VL-2B checkpoint.

## Task: Mobile Agent
<img width="1001" alt="fig1" src="https://github.com/ideachoi337/MobileAgent/blob/main/imgs/fig1.png" />
The mobile agent task is a task where, given a mobile GUI screenshot and a goal to be achieved as input to the model, the model must choose the actions to take in the context of the given screenshot.

The model can choose an appropriate action from among 8 possible actions:

1. Type: Enter a string when text input is enabled.
2. Click: Tap/click a specific location on the screen.
3. Dual_point: Perform a gesture starting from a start point and ending at an end point.
4. Press_back: Press the back button.
5. Press_home: Press the home button.
6. Press_enter: Press the enter button.
7. Task_complete: Indicate that the task is completed.
8. Task_impossible: Indicate that it is impossible to complete the task in the current state.

## Model
<img width="1001" alt="fig2" src="https://github.com/ideachoi337/MobileAgent/blob/main/imgs/fig0.png" />

The open-source MLLM (Multimodal Large Language Model) InternVL2-2B model has been fine-tuned.

***Model ONLY uses an image with instruction to predict next action***

The model input is as follows:
> \<image\>\nPlease provide the bounding box coordinate of the region for this instruction: \<ref\>*What time is it?*\</ref\>

The model output for each action type is as follows:
> * \<ref\>press_home\</ref\>
> * \<ref\>type\</ref\>*some sentences*
> * \<ref\>click\</ref\>\<box\>\[\[*x1*, *y1*, *x2*, *y2*\]\]\</box\>
> * \<ref\>dual_point\</ref\>\<box\>\[\[*x1*, *y1*, *x2*, *y2*\]\]\</box\>\<box\>\[\[*x1*, *y1*, *x2*, *y2*\]\]\</box\>

The coordinates to be tapped are represented in the form of bounding boxes. All coordinates are normalized between 0 and 1000.

## Datasets
* ScreenSpot-v2 dataset:
  - The data consists of screenshot images from PC, web, and mobile environments with instructions and coordinates of corresponding regions.
  - Only used for stage-1 training
* Android-In-The-Wild dataset: 
  - The dataset consists of screenshot images from an Android smartphone environment with instructions and next actions.
  - Because AITW datasets are so large, only 'single' and 'general' class are used for this work.
  - For stage-1 training, only grounding actions(click) is used and all actions are used for stage-2 training.

You can download datasets here:
* [Datasets for stage-1 training: ScreenSpot-v2 + AITW (grounding actions)](https://github.com/ideachoi337/MobileAgent/blob/main/grounding_dataset_gpt.jsonl)
* [Datasets for stage-2 training: AITW (all actions)](https://github.com/ideachoi337/MobileAgent/blob/main/grounding_mobile_dataset_gpt.jsonl)

## Training
> You can train model by script ```GPUS=1 PER_DEVICE_BATCH_SIZE=8 sh shell/finetune.sh```

For fine-tuning, two-stage training was used.
* Stage 1: Trained using only the data corresponding to the "Click" action. This is to ensure that the model outputs in the desired format and to develop its basic GUI understanding abilities. (Total 118418 data)
* Stage 2: Trained using whole the data. This is to enhance the model's ability to choose the appropriate action for a given situation and to improve its understanding of the screenshot image. (Total 170824 data)

The training details are as follows:
* GPU: 1 RTX A6000 48GB (But you can inference with a 16GB GPU)
* Batch size: 16
* num_epoch: 1
* LoRA Training: Freeze backbone and train LoRA layer with r=16 (Also train visual encoder at stage 2 with LoRA layer)
* learning_rate: 4e-5 with cosine scheduler and warmup
* Time taken: 7h + 14h
<img width="800" alt="fig3" src="https://github.com/ideachoi337/MobileAgent/blob/main/imgs/fig2.png" />

## Results (Demo)

**You can see high resolution demo video here: https://github.com/ideachoi337/MobileAgent/blob/main/imgs/demo.mp4**

Instructions in video:
* I want to know about 'Yonsei University'.
* Set an alarm at 6:00 PM
* Install app: LearnUs

<img width="800" alt="search" src="https://github.com/ideachoi337/MobileAgent/blob/main/imgs/search.gif" />
<img width="800" alt="alarm" src="https://github.com/ideachoi337/MobileAgent/blob/main/imgs/alarm.gif" />
<img width="800" alt="store" src="https://github.com/ideachoi337/MobileAgent/blob/main/imgs/store.gif" />

## Cites
> InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks<br>
> How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites<br>
> OS-ATLAS: A Foundation Action Model for Generalist GUI Agents<br>
> Android in the Wild: A Large-Scale Dataset for Android Device Control<br>
