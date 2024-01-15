# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

import torch

from clip import clip
from tqdm import tqdm


COCO_OVD_ALL_CLS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "kite",
    "skateboard",
    "surfboard",
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "bed",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "toothbrush",
]


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


class_only = ["{}"]
single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def build_text_embedding_coco():
    categories = COCO_OVD_ALL_CLS
    run_on_gpu = torch.cuda.is_available()
    templates = multiple_templates

    # clip_model = load_clip_to_cpu("ViT-B/32")
    model, preprocess = clip.load("ViT-B/32", download_root="weights")
    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        for category in tqdm(categories):
            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = ["This is " + text if text.startswith("a") or text.startswith("the") else text for text in texts]
            texts = [category]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        print(f"Finished, get text embeddings of size {all_text_embeddings.T.size()}")
        print("Saving...")
        torch.save(all_text_embeddings.T.cpu(), "ovd_coco_text_embedding.pth")


if __name__ == "__main__":
    build_text_embedding_coco()
