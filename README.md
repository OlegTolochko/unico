# UniCo: Unified Primitive Proxies for Structured Shape Completion

**CVPR 2026**

[![Website](https://img.shields.io/badge/%F0%9F%A4%8D%20Project%20-Website-blue)](https://unico-completion.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2601.00759)

UniCo is a structured shape completion model that, given a partial scan, jointly predicts a complete set of quadratic primitives with geometry, semantics, and inlier membership. The predicted primitives are assembly-ready for surface reconstruction.

<p align="center">
  <a href="https://unico-completion.github.io">
    <img src="assets/teaser.gif" alt="UniCo teaser" width="720px">
  </a>
</p>

## ⚙️ Setup

This repository uses uv for setup. The uv environment installs CUDA-enabled PyTorch wheels from the official PyTorch CUDA 13 index and builds the local CUDA extensions.

1. Clone the repository:

   ```bash
   git clone https://github.com/complete3d/unico.git && cd unico && git lfs pull
   ```

2. Create a uv environment with all dependencies:

   ```bash
   # This creates .venv and builds CUDA extensions
   bash install.sh
   ```


## 🏋️ Training

* Launch training with one of the helper scripts below.

  **DistributedDataParallel (DDP)**:

  ```bash
  # Use only GPU IDs that are present in `nvidia-smi --list-gpus`
  # Override `MASTER_PORT` to avoid collision
  CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh experiment=abcmulti
  ```

  **DataParallel (DP)**:

  ```bash
  CUDA_VISIBLE_DEVICES=0 ./scripts/train_dp.sh experiment=abcmulti
  ```

* TensorBoard logs are written under `output/<experiment>/tensorboard`:

  ```bash
  # Board typically available at http://localhost:6006
  tensorboard --logdir output
  ```

## 🎯 Evaluation

* Run inference with the [provided checkpoint](./ckpt/ckpt-best.pth):

  ```bash
  # Replace device IDs with your own
  CUDA_VISIBLE_DEVICES=0 ./scripts/infer.sh experiment=abcmulti evaluate.mode=easy evaluate.ckpt_path=ckpt/ckpt-best.pth
  ```

* Convert UniCo `.seg` predictions into standard visual files:

  ```bash
  uv run python tools/export_visuals.py evaluation --out-dir evaluation/visual
  ```

  The converter writes colored predicted point clouds as `.ply`, primitive proxy meshes as `.obj`, and combined overlay `.ply` files for 3D viewers. Open `*_overlay.ply` to see primitive faces plus tiny mesh glyphs for the predicted points in one file. If the overlay points are too small or too large in your viewer, rerun with `--point-radius <value>`.

* Primitive assembly uses [PrimFit](https://github.com/xiaowuga/PrimFit) for ABC-multi, and [PolyFit](https://github.com/LiangliangNan/PolyFit), [KSR](https://www.cgal.org/2024/05/29/Kinetic_surface_reconstruction/), and [COMPOD](https://github.com/raphaelsulzer/compod) for plane-only assembly. Sample outputs are provided under `evaluation/`:
  - `VG`: vertex groups for primitives.
  - `SEG`: primitive parameters and memberships.
  - `OBJ`: mesh exports after primitive assembly.

## 📁 Datasets

We provide preprocessed versions of the three datasets used in the paper: [ABC-multi](https://huggingface.co/datasets/chenzhaiyu/abcmulti) (`abcmulti`), [ABC-plane](https://huggingface.co/datasets/chenzhaiyu/abcplane) (`abcplane`), and [BuildingNL](https://huggingface.co/datasets/chenzhaiyu/buildingnl) (`buildingnl`). For quick testing, a small subset of ABC-multi is included under `data/abcmulti/`.

## 🚧 TODOs

- [x] Code release
- [x] Datasets
- [ ] Demo

## 🎓 Citation

If you use UniCo in scientific work, please cite the paper:

<a href="https://arxiv.org/pdf/2601.00759"><img class="image" align="left" width="190px" src="./assets/paper_thumbnail.png"></a>
<a href="https://arxiv.org/abs/2601.00759">[arXiv]</a>&nbsp;&nbsp;<a href="./CITATION.bib">[BibTeX]</a><br>
```bibtex
@article{chen2026unico,
  title={Unified Primitive Proxies for Structured Shape Completion},
  author={Zhaiyu Chen and Yuqing Wang and Xiao Xiang Zhu},
  journal={arXiv preprint arXiv:2601.00759},
  year={2026}
}
```
<br clear="left"/>

## 🙏 Acknowledgements

We thank the authors of [PoinTr](https://github.com/yuxumin/PoinTr) and [PrimFit](https://github.com/xiaowuga/PrimFit) for open-sourcing their great work.
