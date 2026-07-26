# RFCMH

Reference implementation for **Robust Fuzzy Cross-Modal Hashing (RFCMH)**.

## Datasets

The benchmark datasets used in this repository are hosted on Hugging Face:

<https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets>

The first four datasets are **single-label** datasets, and the last three datasets are **multi-label** datasets.

| Dataset | Label type | Download |
| --- | --- | --- |
| INRIA-Websearch | Single-label | [INRIA-Websearch.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/inria-websearch.zip) |
| Wiki | Single-label | [wiki.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/wiki.zip) |
| XMedia | Single-label | [xmedia.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/xmedia.zip) |
| XMediaNet | Single-label | [xmedianet.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/xmedianet.zip) |
| MS-COCO | Multi-label | [MS-COCO.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/MS-COCO.zip) |
| NUS-WIDE | Multi-label | [NUS-WIDE.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/NUS-WIDE.zip) |
| MIRFlickr | Multi-label | [MIRFlickr.zip](https://huggingface.co/datasets/xinxin713/hashing-benchmark-datasets/resolve/main/data/MIRFlickr.zip) |

After downloading, unzip the required dataset archives into your local data directory.

## Usage

Run training with:

```bash
python main.py
```


