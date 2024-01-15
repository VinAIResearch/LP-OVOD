from __future__ import annotations

import mmcv
import numpy as np
import torch
from PIL import Image

import faiss
import json
import logging
from mmdet.models.roi_heads.class_name import COCO_UNSEEN_CLS
from tqdm import tqdm


logger = logging.getLogger()


def retrieve_patches_faiss(text_features):
    text_features = np.ascontiguousarray(text_features.numpy())
    proposals = mmcv.load("lvis_v1/mask_rcnn_lvis_train.pkl")
    proposal_id_map = mmcv.load("lvis_v1/lvis_train_id_map.json")
    clip_dim = 512
    num_proposals = list(len(proposal) for proposal in proposals)
    num_proposals.insert(0, 0)
    num_proposals_cum = np.cumsum(num_proposals)
    clip_features = torch.HalfTensor(
        torch.HalfStorage.from_file("coco_clip_emb_train.pth", shared=False, size=sum(num_proposals) * clip_dim)
    ).view(-1, clip_dim)
    img_ids = list(proposal_id_map.keys())
    all_img_ids = [[] for _ in range(len(text_features))]
    all_proposals = [[] for _ in range(len(text_features))]
    all_embeddings = [[] for _ in range(len(text_features))]
    all_cosine = [[] for _ in range(len(text_features))]
    for i in tqdm(range(0, len(img_ids) // 5000 + 1)):
        batch_features = []
        batch_proposals = []
        batch_img_ids = []
        for img_id in img_ids[5000 * i : 5000 * (i + 1)]:
            id = proposal_id_map[img_id]
            try:
                proposal = np.array(proposals[id])
            except IndexError:
                continue
            valid = proposal[:, -1] >= 0.5
            features = clip_features[num_proposals_cum[id] : num_proposals_cum[id + 1]].float()
            features = torch.nn.functional.normalize(features, dim=-1, p=2).numpy()
            batch_features.append(features[valid].copy())
            batch_proposals.append(proposal[valid].copy())
            batch_img_ids += [img_id] * valid.sum()
        batch_features = np.concatenate(batch_features)
        batch_proposals = np.concatenate(batch_proposals)
        index = faiss.IndexFlatIP(clip_dim)
        index.add(batch_features)
        sim, indices = index.search(text_features, 300)
        selected_proposals = [batch_proposals[ids] for ids in indices]
        selected_embeddings = [batch_features[ids] for ids in indices]
        selected_img_ids = [[batch_img_ids[k] for k in ids] for ids in indices]
        for j in range(len(text_features)):
            all_img_ids[j] += selected_img_ids[j]
            all_proposals[j].append(selected_proposals[j])
            all_embeddings[j].append(selected_embeddings[j])
            all_cosine[j].append(sim[j])
    for j in range(len(text_features)):
        all_proposals[j] = np.concatenate(all_proposals[j])
        all_embeddings[j] = np.concatenate(all_embeddings[j])
        all_cosine[j] = np.concatenate(all_cosine[j])

    saved_dict = {
        "image_ids": all_img_ids,
        "proposals": all_proposals,
        "embeddings": all_embeddings,
        "cosine": all_cosine,
    }
    torch.save(saved_dict, "retrieval/vild_proposal_retrieval_faiss.pkl")


def merge_annotations():
    saved_dict = {
        "image_ids": [[] for i in range(337)],
        "proposals": [[] for i in range(337)],
        "embeddings": [[] for i in range(337)],
        "cosine": [[] for i in range(337)],
    }
    for i in range(1, 4):
        data = torch.load(f"lvis_v1/mask_proposal_retrieval_faiss_part{i}_06.pkl")
        for j in tqdm(range(337)):
            saved_dict["image_ids"][j] += data["image_ids"][j]
            saved_dict["proposals"][j].append(data["proposals"][j])
            saved_dict["embeddings"][j].append(data["embeddings"][j])
            saved_dict["cosine"][j].append(data["cosine"][j])

    for j in tqdm(range(337)):
        saved_dict["proposals"][j] = np.concatenate(saved_dict["proposals"][j])
        saved_dict["embeddings"][j] = np.concatenate(saved_dict["embeddings"][j])
        saved_dict["cosine"][j] = np.concatenate(saved_dict["cosine"][j])

    for j in range(337):
        print(f"{j}: {saved_dict['proposals'][j].shape}")

    torch.save(saved_dict, "lvis_v1/mask_lvis_proposal_retrieval_faiss06.pkl")


def build_novel_dict_faiss_json(num_samples):
    proposals = torch.load("voc/voc_vild_proposal_retrieval_faiss.pkl")
    text_embeddings = torch.load("voc/ovd_voc_vild_text_embedding.pth", map_location="cpu").float()
    text_embeddings = torch.nn.functional.normalize(text_embeddings).numpy()
    text_embeddings = np.ascontiguousarray(text_embeddings)
    saved_dict = {id: [] for id in range(1, len(text_embeddings) + 1)}
    annotations = []
    images = []
    image_id_set = set()
    for i, id in enumerate(range(1, len(text_embeddings) + 1)):
        print(id)
        candidate_img_ids = proposals["image_ids"][i]
        candidate_proposals = proposals["proposals"][i]
        candidate_embeddings = proposals["embeddings"][i]

        candidate_embeddings = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, ord=2, axis=-1, keepdims=True
        )

        index = faiss.IndexFlatIP(512)
        index.add(candidate_embeddings)
        sim, indices = index.search(text_embeddings[i : i + 1], 300)
        sim = sim.reshape([-1])
        indices = indices.reshape([-1])
        selected_proposals = candidate_proposals[indices]
        selected_embedding = candidate_embeddings[indices]
        selected_img_ids = [candidate_img_ids[ids] for ids in indices]
        image_set = set()
        count = 0
        for k in range(len(selected_img_ids)):
            if count == num_samples:
                break
            # if selected_img_ids[k] not in image_set:
            image_set.add(selected_img_ids[k])
            img_info = {
                "img_id": selected_img_ids[k],
                "proposal": selected_proposals[k],
                "features": selected_embedding[k],
                "cosine": sim[k],
            }
            saved_dict[id].append(img_info)
            count += 1
        # saved_dict[id] = [{"img_id": selected_img_ids[k], "proposal": selected_proposals[k], "features": selected_embedding[k], "cosine": sim[k]} \
        #                   for k in range(len(indices))]
    # with open("object365/zhiyuan_objv2_val.json", "rb") as f:
    #    data = json.load(f)
    categories = [{"name": name, "id": id + 1} for id, name in enumerate(COCO_UNSEEN_CLS)]
    # categories = data["categories"]
    ann_id = 1
    print(saved_dict.keys())
    for id in tqdm(saved_dict.keys()):
        category_id = id
        for ann in saved_dict[id]:
            img_id = int(ann["img_id"])
            proposal = ann["proposal"]
            left, top, right, bottom = proposal[:4]
            score = proposal[-1]
            cosine = ann["cosine"]
            if img_id not in image_id_set:
                image_id_set.add(img_id)
                pil_image = Image.open(f"data/coco/train2017/{str(img_id).zfill(12) + '.jpg'}")
                width = pil_image.width
                height = pil_image.height
                images.append(
                    {"file_name": str(img_id).zfill(12) + ".jpg", "width": width, "height": height, "id": img_id}
                )
            width = right - left
            height = bottom - top
            area = float(width * height)
            ann = {
                "segmentation": [],
                "area": area,
                "bbox": [float(left), float(top), float(width), float(height)],
                "image_id": img_id,
                "id": ann_id,
                "category_id": category_id,
                "iscrowd": 0,
                "score": float(score),
                "cosine": float(cosine),
            }
            ann_id += 1
            annotations.append(ann)
    print(len(annotations))
    json_annotations = {"annotations": annotations, "images": images, "categories": categories}
    with open(f"retrieval/voc_vild_proposal{num_samples}.json", "w") as f:
        json.dump(json_annotations, f)


if __name__ == "__main__":
    embeddings = torch.load("ovd_coco_text_embedding.pth")
    embeddings = embeddings.float().cpu()
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
    build_novel_dict_faiss_json(100)
