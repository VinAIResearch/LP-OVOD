import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

import argparse
import clip
import time
from mmdet.datasets import CocoDataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import os


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class CocoCroppedProposals(CocoDataset):
    n_px = 224
    clip_transform = Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            # _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    scales = [1.0, 1.5]

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=False,
        proposal_id_map=None,
    ):
        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
            proposal_id_map,
        )
        self.num_proposals_per_img = list(len(proposal) for proposal in self.proposals)
        self.num_proposals_per_img.insert(0, 0)
        self.num_proposals_per_img_cum = np.cumsum(self.num_proposals_per_img)

    def __len__(self):
        return self.num_proposals_per_img_cum[-1]

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        img_id = np.searchsorted(self.num_proposals_per_img_cum, idx, side="right") - 1
        proposal_id = idx - self.num_proposals_per_img_cum[img_id]

        img_info = self.data_infos[img_id]
        if self.img_prefix is not None:
            filename = os.path.join(self.img_prefix, img_info["filename"])
        else:
            filename = img_info["filename"]

        image = Image.open(filename)
        proposal = self.proposals[img_id][proposal_id][:4]
        bboxes, bboxes15 = self.get_bboxes_and_bboxes15(proposal, img_info["height"], img_info["width"])
        img, img15 = image.crop(bboxes), image.crop(bboxes15)
        img = self.clip_transform(img)
        img15 = self.clip_transform(img15)

        return img, img15, idx

    def get_bboxes_and_bboxes15(self, bboxes, h, w):
        def box_coord_2int(box):
            box = (
                np.floor(box[0] - 0.001),
                np.floor(box[1] - 0.001),
                np.ceil(box[2] + 0.001),
                np.ceil(box[3] + 0.001),
            )
            return (
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], w),
                min(box[3], h),
            )

        assert len(bboxes) == 4
        bboxes15 = (
            1.25 * bboxes[0] - 0.25 * bboxes[2],
            1.25 * bboxes[1] - 0.25 * bboxes[3],
            1.25 * bboxes[2] - 0.25 * bboxes[0],
            1.25 * bboxes[3] - 0.25 * bboxes[1],
        )
        bboxes = box_coord_2int(bboxes)
        bboxes15 = box_coord_2int(bboxes15)
        return bboxes, bboxes15

    def _set_group_flag(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Extract COCO embeddings")
    parser.add_argument("--data_root", default="data/coco/", help="data root")
    parser.add_argument("--split", default="train", help="data split")
    parser.add_argument("--proposal_file", default="path_to_proposal", help="path to pre-computed proposals")
    parser.add_argument("--clip_root", default="", help="clip model path")
    parser.add_argument("--num_workers", default=1, type=int, help="num workers per gpu")
    parser.add_argument("--batch_size", default=100, type=int, help="batch size per gpu")
    parser.add_argument("--save_path", default="embeddings.pth", help="path to save output")
    parser.add_argument("--")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)

    data_root = args.data_root
    dataset = CocoCroppedProposals(
        pipeline=[],
        ann_file=os.path.join(data_root, f"annotations/instances_{args.split}2017.json"),
        proposal_file=args.proposal_file,
        img_prefix=data_root + f"{args.split}2017",
    )

    clip_model, _ = clip.load("ViT-B/32", device=device_id, download_root=args.clip_root)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)

    embeddings = torch.HalfTensor(
        torch.HalfStorage.from_file(args.save_path, shared=True, size=len(dataset) * clip_model.visual.output_dim)
    ).reshape(len(dataset), -1)
    start = time.time()
    with torch.no_grad():
        for step, (img, img15, idx) in enumerate(dataloader):
            if rank == 0:
                if step % 1000 == 0:
                    print(f"Step: {step}/{len(dataloader)}, Time: {(time.time() - start):.2f}s")
            clip_image_features = clip_model.encode_image(img)
            clip_image_features15 = clip_model.encode_image(img15)

            clip_image_features_single = clip_image_features + clip_image_features15
            clip_image_features = F.normalize(clip_image_features_single, p=2, dim=1)

            embeddings[idx] = clip_image_features.cpu()


if __name__ == "__main__":
    main()
