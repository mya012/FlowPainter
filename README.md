# [ECCV 2026] FlowPainter: Inpainting Optical Flow via Confidence-Guided Completion

FlowPainter is a confidence-guided diffusion framework for optical flow estimation that reformulates flow prediction as a soft inpainting-style refinement process. It leverages a lightweight confidence-aware network to provide a rough flow prior and identify reliable motion regions, enabling the diffusion model to focus its refinement on uncertain and challenging areas.

The repository provides the implementation of FlowPainter for training, evaluation, and benchmark submission generation on FlyingChairs, FlyingThings3D, Sintel, and KITTI. The core model components are maintained in `core/`, while training, evaluation, checkpoint management, and submission utilities are organized under the `flow_painter` package.

## Authors

**Yuang Meng**<sup>1,3</sup>, **Chenyang Wu**<sup>1</sup>, **Xianshun Liu**<sup>1,3</sup>, **Chun-Le Guo**<sup>1,*</sup>,
**Zichen Liang**<sup>1</sup>, **Lina Lei**<sup>1</sup>, **Jie Liang**<sup>3</sup>, **Hui Zeng**<sup>3</sup>, **Chongyi Li**<sup>1</sup>, **Lei Zhang**<sup>2,3</sup>

<sup>1</sup> VCIP, CS, Nankai University <sup>2</sup> The Hong Kong Polytechnic University <sup>3</sup> OPPO Research Institute

## Installation

Create a Python environment with PyTorch and the common optical-flow dependencies:

```bash
git clone https://github.com/mya012/FlowPainter.git
cd FlowPainter

conda create -n FlowPainter python=3.11.2
conda activate FlowPainter

pip install -r requirements.txt
```

## Data

Dataset paths can be passed through CLI arguments or environment variables. The default fallback is a relative `data/` directory under the project root.

## Checkpoints

Place model checkpoints in `weights/` or pass explicit paths with:

```bash
--model path/to/model.pth
--base_model_ckpt path/to/base-flow-prior.pth
```

By default, the auxiliary flow prior is loaded from:

```text
weights/100000_raft-sintel.pth
```

You can download the model from [this Google Drive link](https://drive.google.com/drive/folders/1Z0lJlPvUkNy5OTcwCgJKYSXaqOZ8_Ztb?usp=drive_link).

## Training and Evaluation

The compact shell recipe follows the staged training schedule:

```bash
bash train.sh
```

You can also run one stage manually:

```bash
python -m flow_painter.cli.train \
  --name fd-kitti \
  --stage kitti \
  --validation kitti \
  --restore_ckpt checkpoints/fd-sintel.pth \
  --base_model_ckpt weights/100000_raft-sintel.pth \
  --gpus 0 1 2 3 \
  --num_steps 15000 \
  --batch_size 6 \
  --lr 0.0001 \
  --image_size 288 960 \
  --wdecay 0.00001 \
  --gamma 0.85
```

The compact shell recipe follows the staged evaluation schedule:

```bash
bash eval.sh
```

Run validation explicitly:

```bash
python -m flow_painter.cli.evaluate \
  --model weights/5000_fd-kitti.pth \
  --dataset kitti \
  --task validation \
  --base_model_ckpt weights/100000_raft-sintel.pth
```

## Acknowledgement

The code is built based on [FlowDiffuser](https://github.com/LA30/FlowDiffuser), [MaskFlowNet](https://github.com/microsoft/MaskFlownet), and [RAFT](https://github.com/princeton-vl/RAFT). We thank the authors for their contributions.

