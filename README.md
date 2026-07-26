<h1 align="center">
  Learning with Admissibility:<br>
  Robust Fuzzy Hashing for Cross-Modal Retrieval with Noisy Labels
</h1>

<p align="center">
  <b>RFCMH</b>
</p>

<p align="center">
  Xincheng Sun<sup>1,2,*</sup>,
  Ruitao Pu<sup>3,*</sup>,
  Guangsi Shi<sup>4</sup>,
  Zhenwen Ren<sup>5</sup>,
  Peng Hu<sup>3</sup>,
  Yuan Sun<sup>1,&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup>National Key Laboratory for Fundamental Algorithms and Models for Engineering Simulation, Sichuan University<br>
  <sup>2</sup>Department of Computer Science and Technology, Heilongjiang University<br>
  <sup>3</sup>School of Computer Science, Sichuan University<br>
  <sup>4</sup>Corporate Research Center, Midea Group<br>
  <sup>5</sup>Southwest University of Science and Technology
</p>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup> Corresponding author
</p>

<p align="center">
  <b>ICML 2026 Spotlight &middot; Proceedings of the 43rd International Conference on Machine Learning</b>
</p>

<p align="center">
  <a href="https://github.com/XinchengSun/RFCMH">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-Hugging%20Face-FFD21E" alt="Hugging Face Data">
  </a>
  <img src="https://img.shields.io/badge/ICML%202026-Spotlight-B31B1B" alt="ICML 2026 Spotlight">
</p>

<p align="center">
  <a href="#-news">News</a> |
  <a href="#-highlights">Highlights</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-preparation">Preparation</a> |
  <a href="#-training">Training</a> |
  <a href="#-citation">Citation</a>
</p>

RFCMH is a robust fuzzy cross-modal hashing framework for retrieval under noisy supervision. It introduces fuzzy set theory to endow noisy labels with admissibility, then uses Fuzzy Admissibility Refinement (FAR) and Dual-Granularity Structural Alignment (DGSA) to obtain reliable discriminative supervision and stable cross-modal representations.

## 📢 News

- RFCMH is selected as an **ICML 2026 Spotlight**.
- Code and benchmark dataset links are now available.

## ✨ Highlights

- Robust fuzzy supervision for cross-modal hashing with noisy labels.
- Unified image-to-text and text-to-image retrieval evaluation.
- Support for symmetric and asymmetric label noise settings.
- Public benchmark archives hosted on Hugging Face for easier reproduction.

## 🤗 Datasets

All benchmark archives are hosted on Hugging Face:

<p align="center">
  <a href="https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets">
    <b>🤗 Download Datasets from Hugging Face</b>
  </a>
</p>

The table follows the benchmark order used in this repository: the first four datasets are **single-label** datasets, and the last three datasets are **multi-label** datasets.

| Dataset | Label type | Archive | Download |
| --- | --- | --- | --- |
| INRIA-Websearch | Single-label | `data/inria-websearch.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/inria-websearch.zip) |
| Wiki | Single-label | `data/wiki.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/wiki.zip) |
| XMedia | Single-label | `data/xmedia.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/xmedia.zip) |
| XMediaNet | Single-label | `data/xmedianet.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/xmedianet.zip) |
| MS-COCO | Multi-label | `data/MS-COCO.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/MS-COCO.zip) |
| NUS-WIDE | Multi-label | `data/NUS-WIDE.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/NUS-WIDE.zip) |
| MIRFlickr | Multi-label | `data/MIRFlickr.zip` | [Download](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/MIRFlickr.zip) |

You can also download all archives with the Hugging Face CLI:

```bash
hf download xinxin713/hashing-benchmark-datasets --repo-type dataset --include "data/*.zip"
```

After downloading, unzip the required archives and place the extracted dataset files under your local dataset directory.

## 🛠 Preparation

Install the required Python packages in your environment:

```bash
pip install -U pip
pip install torch torchvision scipy numpy
```

Prepare the datasets by downloading the archives above and extracting the `.mat` or `.h5` files.

## 🚀 Training

Run RFCMH with:

```bash
python main.py --dataset INRIA-Websearch --data_path /path/to/INRIA-Websearch.mat
```

Common options include:

```bash
python main.py \
  --dataset INRIA-Websearch \
  --data_path /path/to/INRIA-Websearch.mat \
  --bit 128 \
  --noisy_ratio 0.8 \
  --noise_mode sym \
  --GPU 0
```

## 📖 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{sun2026learning,
  title={Learning with Admissibility: Robust Fuzzy Hashing for Cross-Modal Retrieval with Noisy Labels},
  author={Sun, Xincheng and Pu, Ruitao and Shi, Guangsi and Ren, Zhenwen and Hu, Peng and Sun, Yuan},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={306},
  year={2026},
  publisher={PMLR}
}
```
