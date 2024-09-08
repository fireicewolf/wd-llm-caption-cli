# WD Joy Caption Cli
A Python base cli tool for caption images with [WD series](https://huggingface.co/SmilingWolf) and [joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha) models.
## Introduce
If you want to caption a training datasets for Image generation model(Stable Diffusion, Flux, Kolors or others)
This tool can make a caption with danbooru style tags or a nature language description.
## Example
<img alt="DEMO_her.jpg" src="DEMO/DEMO_her.jpg" width="300" height="400"/>

#### WD Caption
Use wd-eva02-large-tagger-v3
```text
1girl, solo, long hair, breasts, looking at viewer, smile, blue eyes, blonde hair, medium breasts, white hair, ass, looking back, blunt bangs, from behind, english text, lips, night, building, science fiction, city, railing, realistic, android, cityscape, joints, cyborg, robot joints, city lights, mechanical parts, cyberpunk
```
#### Joy Caption
Default LLama3.1 8B, no quantization
```text
This is a digitally rendered image, likely created using advanced CGI techniques, featuring a young woman with a slender, athletic build and long, straight platinum blonde hair with bangs. She has fair skin and a confident, slightly playful expression. She is dressed in a futuristic, form-fitting suit that combines sleek, metallic armor with organic-looking, glossy black panels. The suit accentuates her curvaceous figure, emphasizing her ample breasts and hourglass waist. She stands on a balcony with a red railing, overlooking a nighttime cityscape with a prominent, illuminated tower in the background. The city is bustling with lights from various buildings, creating a vibrant, urban atmosphere. The text at the top of the image reads "PUBLISHED ON 2024.07.30," followed by "AN AIGC WORK BY DUKG" and "GENERATED BY STABLE DIFFUSION." Below, there are smaller texts indicating the artist's name and the studio where the image was created. The overall style is high-tech and futuristic, with a blend of cyberpunk and anime aesthetics, highlighting the intersection of human and machine elements in a visually striking and provocative manner.
```
#### Joy Caption with WD
Use wd-eva02-large-tagger-v3 and LLama3.1 8B, no quantization.
WD tags used in LLama3.1 user prompt.
```text
The image is a high-resolution photograph featuring a young woman with long, platinum blonde hair and blue eyes. She is dressed in a sleek, form-fitting white and black bodysuit that resembles a futuristic cyborg suit, with visible mechanical joints and metallic textures. Her physique is slender and toned, with a noticeable emphasis on her hips and buttocks. She is standing on a red railing, with a cityscape in the background, including a prominent tower with a red antenna. The night sky is filled with twinkling city lights, creating a vibrant, cyberpunk atmosphere. The text at the top reads "PUBLISH ON 2024.07.30" and "An IG work by DukeG" at the bottom. The overall style is realistic, with a focus on modern, high-tech aesthetics.
```
## Model source
Hugging Face are original sources, modelscope are pure forks from Hugging Face(Because Hugging Face was blocked in Some place).
### WD Capiton models
|            Model             |                                Hugging Face Link                                |                                     ModelScope Link                                     |
|:----------------------------:|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
|   wd-eva02-large-tagger-v3   |   [Hugging Face](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)   |   [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-eva02-large-tagger-v3)   |
|    wd-vit-large-tagger-v3    |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3)    |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-vit-large-tagger-v3)    |
|     wd-swinv2-tagger-v3      |     [Hugging Face](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)      |     [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-swinv2-tagger-v3)      |
|       wd-vit-tagger-v3       |       [Hugging Face](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3)       |       [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-vit-tagger-v3)       |
|    wd-convnext-tagger-v3     |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3)     |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-convnext-tagger-v3)     |
|    wd-v1-4-moat-tagger-v2    |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)    |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-moat-tagger-v2)    |
|   wd-v1-4-swinv2-tagger-v2   |   [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)   |   [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-swinv2-tagger-v2)   |
| wd-v1-4-convnextv2-tagger-v2 | [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2) | [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-convnextv2-tagger-v2) |
|    wd-v1-4-vit-tagger-v2     |    [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)     |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-vit-tagger-v2)     |
|  wd-v1-4-convnext-tagger-v2  |  [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)  |  [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-convnext-tagger-v2)  |
|      wd-v1-4-vit-tagger      |      [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger)      |      [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-vit-tagger)      |
|   wd-v1-4-convnext-tagger    |   [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger)    |   [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-convnext-tagger)    |
|      Z3D-E621-Convnext       |         [Hugging Face](https://huggingface.co/toynya/Z3D-E621-Convnext)         |      [ModelScope](https://www.modelscope.cn/models/fireicewolf/Z3D-E621-Convnext)       |
### Joy Caption models
|               Model               |                                 Hugging Face Link                                 |                                      ModelScope Link                                       |
|:---------------------------------:|:---------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
|       joy-caption-pre-alpha       |  [Hugging Face](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)   |      [ModelScope](https://www.modelscope.cn/models/fireicewolf/joy-caption-pre-alpha)      |
| siglip-so400m-patch14-384(Google) |      [Hugging Face](https://huggingface.co/google/siglip-so400m-patch14-384)      |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/siglip-so400m-patch14-384)    |
|         Meta-Llama-3.1-8B         |        [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)        |        [ModelScope](https://www.modelscope.cn/models/fireicewolf/Meta-Llama-3.1-8B)        |
|  Llama-3.1-8B-Lexi-Uncensored-V2  | [Hugging Face](https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2) | [ModelScope](https://www.modelscope.cn/models/fireicewolf/Llama-3.1-8B-Lexi-Uncensored-V2) |

## TO-DO
make a simple ui by Jupyter widget(When my lazy cancer cured😊)
## Installation
Python 3.10 works fine.

Open a shell terminal and follow below steps:
```shell
# Clone this repo
git clone https://github.com/fireicewolf/wd-joy-caption-cli.git
cd wd-joy-caption-cli

# create a Python venv
python -m venv .venv
.\venv\Scripts\activate

# Install torch
# Install torch base on your GPU driver. e.g.
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
 
# Base dependencies, models for inference will download via python request libs.
# For WD Caption
pip install -U -r requirements_wd.txt
# If you want load WD models with GPU.
# For CUDA 11.8
pip install -U -r requirements_onnx_cu118.txt
# For CUDA 12.X
pip install -U -r requirements_onnx_cu12x.txt

# For Joy Caption
pip install -U -r requirements_joy.txt

# If you want to download or cache model via huggingface hub, install this.
pip install -U -r requirements_huggingface.txt

# If you want to download or cache model via modelscope hub, install this.
pip install -U -r requirements_modelscope.txt
```
## Simple usage
Default will use both wd and joy caption to caption images,  
Joy caption used Meta Llama 3.1 8B, on Hugging Face it is a gated models, so you need get access on Hugging Face first.  
Then add `HF_TOKEN` to your environment variable.  

Windows Powershell
```shell
$Env:HF_TOKEN="yourhftoken"
```
Windows CMD
```shell
set HF_TOKEN="yourhftoken"
```
Mac or Linux shell
```shell
export HF_TOKEN="yourhftoken"
```
In python script
```python
import os

os.environ["HF_TOKEN"]="yourhftoken"
```
__Make sure your python venv has been activated first!__
```shell
python caption.py your_datasets_path
```
To run with more options, You can find help by run with this or see at [Options](#options)
```shell
python caption.py -h
```
##  <span id="options">Options</span>
<details>
    <summary>Advance options</summary>
`data_path`

path where your datasets place

`--recursive`

Will include all support images format in your input datasets path and its sub-path.

`--log_level`

set log level[`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`], default is `INFO`

`--save_logs`

save log file.
logs will be saved at same level path with `data_path`. 
e.g., Your input `data_path` is `/home/mydatasets`, your logs will be saved in `/home/`,named as `mydatasets_xxxxxxxxx.log`(x means log created date.),

`--model_site`

download model from model site huggingface or modelscope, default is "huggingface".

`--models_save_path`

path to save models, default is `models`(Under wd-joy-caption-cli)

`--use_sdk_cache`

use sdk\'s cache dir to store models. if this option enabled, `--models_save_path` will be ignored.

`--download_method`

download models via SDK or URL, default is `SDK`(If download via SDK failed, will auto retry with URL).

`--force_download`

force download even file exists.

`--skip_download`


skip download if file exists.

`--wd_config`

configs json for wd tagger models, default is `default_wd.json`

`--wd_model_name`

wd tagger model name will be used for caption inference, default is `wd-swinv2-v3`.

`--wd_force_use_cpu`

force use cpu for wd models inference.

`--wd_caption_extension`

extension for wd captions files while `caption_method` is `both`, default is `.wdcaption`.

`--wd_remove_underscore`

replace underscores with spaces in the output tags.
e.g., `hold_in_hands` will be `hold in hands`.

`--wd_undesired_tags`

comma-separated list of undesired tags to remove from the wd captions.

`--wd_tags_frequency`

Show frequency of tags for images.

`--wd_threshold`

threshold of confidence to add a tag, default value is `0.35`.

`--wd_general_threshold`

threshold of confidence to add a tag from general category, same as `--threshold` if omitted.

`--wd_character_threshold`

threshold of confidence to add a tag for character category, same as `--threshold` if omitted.

`--wd_add_rating_tags_to_first`

Adds rating tags to the first.

`--wd_add_rating_tags_to_last`

Adds rating tags to the last.

`--wd_character_tags_first`

Always put character tags before the general tags.

`--wd_always_first_tags`

comma-separated list of tags to always put at the beginning, e.g. `1girl,solo`

`--wd_caption_separator`

Separator for captions(include space if needed), default is `, `.

`--wd_tag_replacement`

tag replacement in the format of `source1,target1;source2,target2; ...`.
Escape `,` and `;` with `\\`. e.g. `tag1,tag2;tag3,tag4

`--wd_character_tag_expand`

expand tag tail parenthesis to another tag for character tags.
e.g., `character_name_(series)` will be expanded to `character_name, series`.

`--joy_config`

config json for Joy Caption models, default is `default_joy.json`

`--joy_model_name`

model name for inference, default is `Joy-Caption-Pre-Alpha`

`--joy_use_cpu`

load joy models use cpu.

`--joy_llm_dtype`

choice joy llm load dtype[`fp16`, `bf16`], default is `fp16`.

`--joy_llm_qnt`

Enable quantization for joy llm [`none`,`4bit`, `8bit`]. default is `none`.

`--joy_caption_extension`

extension of caption file, default is `.txt`

`--joy_read_wd_caption`

joy will read wd caption for inference. Only effect when `caption_method` is `joy`

`--joy_caption_without_wd`

joy will not read wd caption for inference.Only effect when `caption_method` is `both`

`--joy_user_prompt`

user prompt for caption.

`--joy_temperature`

temperature for joy LLM model, default is `0.5`.

`--joy_max_tokens`

max tokens for joy LLM model output, default is `300`.

`--caption_method`

method for caption[`both`, `wd`, `joy`],select wd or joy models, or both of them to caption, 
default is `both`.

`--run_method`

running method for wd+joy caption[`sync`, `queue`], need `caption_method` set to `both`.
if `sync`, image will caption with wd models,
then caption with joy models while wd captions in joy user prompt.
if `queue`, all images will caption with wd models first,
then caption all of them with joy models while wd captions in joy user prompt.
default is `sync`.

`--image_size`

resize image to suitable, default is `1024`.

`--not_overwrite`

not overwrite caption file if exists.

`--custom_caption_save_path`

custom caption file save path.
</details>

## Credits
Base on [SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) and [joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)
Without their works(👏👏), this repo won't exist.
