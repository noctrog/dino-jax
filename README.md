# DINO Jax

A reimplementation of the original DINO paper in Flax/NNX.

# Installation

It is recommended to use [`uv`](https://github.com/astral-sh/uv), then you can install the python environment with:

```bash
uv sync
```

# Training

You will first need to manually download `imagenet-1k` and store both
the `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` it in
`~/tensorflow_datasets/downloads/manual`.

Simply run:

```bash
uv run dino/train.py
```

This will train a ViT-S on `imagenet-1k` for 100 epochs using data parallelism
across all visible devices. On a workstation with x8 RTX 4090, this takes around
10h. Model checkpoints are enabeld by default. If you are using `wandb` they
will be saved in the run folder (i.e. `wandb/run-<date>_<id>/files`), otherwise
they will be stored in the `outputs` folder under the project root. You can
restore a training session by specifying the checkpoint folder with `--restore`.
To view all flags, just call the script with `--help`.

# Evaluation

To evaluate a trained model using k-NN, run:

```bash
uv run dino/eval_knn.py --ckpt wandb/run-<date>_<id>/files
```

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Citation

If you use this code in your research, please consider citing the original DINO paper:

```bibtex
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J'egou, Herv'e and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

# Acknowledgements

- This project is a reimplementation of the original [DINO](https://github.com/facebookresearch/dino) paper.
- Built with [Flax](https://github.com/google/flax), [Grain](https://github.com/google/grain) and [Albumentations](https://albumentations.ai/docs/).
