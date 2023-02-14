'''
Following this tutorial
'''
import torch 
from lavis.datasets.builders import load_dataset
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.datasets.builders import dataset_zoo


dataset_names = dataset_zoo.get_names()
print(dataset_names)
# ['aok_vqa', 'coco_caption', 'coco_retrieval', 'coco_vqa', 'conceptual_caption_12m',
#  'conceptual_caption_3m', 'didemo_retrieval', 'flickr30k', 'imagenet', 'laion2B_multi',
#  'msrvtt_caption', 'msrvtt_qa', 'msrvtt_retrieval', 'msvd_caption', 'msvd_qa', 'nlvr',
#  'nocaps', 'ok_vqa', 'sbu_caption', 'snli_ve', 'vatex_caption', 'vg_caption', 'vg_vqa']
print(len(dataset_names))
# 23



coco_dataset = load_dataset("coco_caption")

print(coco_dataset.keys())
# dict_keys(['train', 'val', 'test'])

print(len(coco_dataset["train"]))
# 566747

print(coco_dataset["train"][0])


# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load sample image
# raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
img_name = "merlion.png"
raw_image = Image.open(f"../docs/_static/{img_name}").convert("RGB")
print(img_name)
print(raw_image)



# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# generate caption
caption = model.generate({"image": image})
print(caption)
# ['a large fountain spewing water into the air']
print("reached end")