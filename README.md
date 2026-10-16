# OLHWG-PageSynth

## 项目简介

本项目是对 ICLR 2025 论文 **《Decoupling Layout from Glyph in Online Chinese Handwriting Generation》** 的复现与工程化改造，基于 **CASIA-OLHWDB 在线手写数据集（1020 位书写者）** 实现。

论文提出分层生成框架：布局生成器（LSTM）预测每个字的边界框，字符合成器（1D U-Net + DDPM）在框内生成对应风格的轨迹。本项目在此基础上完成了四项工作：

1. **完整复现**：跑通论文的单字生成 + 行布局生成 + 整页拼装流水线
2. **数据重建**：从 `.pot` 原始文件重建完整的 1020 位作者数据，修复原始 `.npy` 数据集中 300 位作者数据不完整的问题
3. **页面级风格**：从 `.wptt` 提取每位作者的行距、行首缩进、行倾斜等页面级排版统计，实现"页级"而非"行级"的风格模仿
4. **批处理推理**：将逐字推理改为一行内批处理，单页耗时从数小时压缩到 400~470 秒

## 数据集

使用 CASIA-OLHWDB 的两个子集，二者由**同一批 1020 位书写者**产生：

用途 | 数据集 | 原始格式 | 本项目使用
---|---|---|---
单字风格 | CASIA-OLHWDB **1.0-1.2** | `.pot` | `datas_full.npy`
页面级排版 | CASIA-OLHWDB **2.0-2.2** | `.wptt` | `page_stats.npy`
行布局 | CASIA-OLHWDB **2.0-2.2** | `.npy` | `all_datas.npy`

**关于数据重建**：HuggingFace 上的预处理版本 `datas.npy` 中，DB1.2 的 300 位作者数据不完整（每人仅约 260 字）。本项目使用官方 `.pot` 文件重建，1020 位作者每人 2460~4036 字。

数据集版权归中国科学院自动化研究所（NLPR）所有，请仅用于学术研究。

## 项目结构

```
OLHWG-PageSynth/
├── configs/
│   ├── default.yaml
│   └── author_split.yaml           # 01_prepare_data.py 自动生成
├── data/
│   ├── metadata.csv
│   ├── raw/                        # .pot / .wptt 原始文件
│   └── processed/
│       ├── datas_full.npy          # ★ 重建的 1020 位作者单字数据
│       ├── writer_id_map.json      # ★ 连续索引 ↔ 原编号映射
│       ├── all_datas.npy           # 行级数据集
│       ├── dictionary.npy          # 行字典（3057 类）
│       ├── mydict.npy              # 单字字典（4055 类）
│       ├── char_boxes.npy          # 字符边界框先验
│       ├── char_lengths.npy        # 字符平均长度
│       ├── page_stats.npy          # ★ 页面级排版统计
│       └── style_features/         # 1020 位作者的风格特征
├── checkpoints/pretrained/
│   ├── unetddpm450000.pth
│   ├── style_encoder450000.pth
│   └── boxlstm3000.pth
├── src/
│   ├── models/                     # 神经网络定义
│   ├── datasets/                   # 数据加载（含 .pot/.wptt 解析）
│   ├── pipeline/                   # 风格提取、行拼装
│   ├── evaluation/                 # 评估指标
│   └── utils/                      # 轨迹处理、可视化
├── scripts/
│   ├── 00_build_char_boxes.py
│   ├── 00b_build_char_lengths.py
│   ├── 00c_build_page_stats.py     # 从 .wptt 提取页面排版统计
│   ├── 00d_rebuild_datas_from_pot.py  # 从 .pot 重建完整数据集
│   ├── 01_prepare_data.py
│   ├── 04_infer_single_page.py     # 单页生成（批处理版）
│   ├── 05_run_batch_1020.py        # 批量生成（并行 + 断点续传）
│   └── ...
├── outputs/
│   ├── pages/                      # 单作者生成结果
│   └── fake_pages/                 # 批量生成结果
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib tqdm pyyaml fastdtw scikit-learn opencv-python
```

### 2. 数据准备

**从 CASIA 官网下载** `.pot` / `.wptt` 文件，放置于 `data/raw/`：

```
data/raw/
├── Pot1.0Train/, Pot1.0Test_2/
├── Pot1.1Train_2/, Pot1.1Test_2/
├── Pot1.2Train_2/, Pot1.2Test/
├── WPTT2.0-Train/, WPTT2.0-Test/
├── WPTT2.1-Train/, WPTT2.1-Test/
└── WPTT2.2-Train/, WPTT2.2-Test/
```

**从 HuggingFace 下载**预训练权重到 `checkpoints/pretrained/`：

```bash
wget https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/unetddpm450000.pth
wget https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/style_encoder450000.pth
# boxlstm3000.pth 从原仓库获取或自行训练
```

**从 HuggingFace 下载**辅助 `.npy` 到 `data/processed/`：

```bash
wget https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/all_datas.npy
wget https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/dictionary.npy
wget https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/mydict.npy
wget https://huggingface.co/datasets/Immortalman12/OLHWD/resolve/main/test_datas.npy
```

### 3. 数据预处理（一次性，约 30 分钟）

```bash
# 从 .pot 重建完整的 1020 位作者单字数据
python scripts/00d_rebuild_datas_from_pot.py

# 生成字符边界框先验
python scripts/00_build_char_boxes.py

# 生成字符平均长度
python scripts/00b_build_char_lengths.py

# 从 .wptt 提取页面级排版统计
python scripts/00c_build_page_stats.py

# 提取并缓存 1020 位作者的风格特征
python scripts/01_prepare_data.py --config configs/default.yaml
```

### 4. 单页生成

```bash
python scripts/04_infer_single_page.py --config configs/default.yaml --author_id 0
```

输出到 `outputs/pages/author_0000/generated_page.png`。

**文本来源**：默认从该作者的测试页 `.wptt` 提取真实文本（含换行和段首缩进）。也可用 `--text` 覆盖：

```bash
python scripts/04_infer_single_page.py --config configs/default.yaml \
    --author_id 0 \
    --text "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
```

### 5. 批量生成

```bash
# 测 2 位作者，验证流程
python scripts/05_run_batch_1020.py --n_workers 1 --author_ids 0,1

# 全量 1020 位作者
python scripts/05_run_batch_1020.py --n_workers 2
```

**参数说明**：
- `--n_workers`：并行进程数。GPU 已饱和时增加无效，建议 1~2
- `--author_ids`：逗号分隔的连续索引，不填则跑全部 1020 位
- **断点续传**：已生成的作者自动跳过

结果保存到 `outputs/fake_pages/author_XXXX.npy` + `.json`。

## 推理流程说明

`04_infer_single_page.py` 不依赖 `.wptt` 之外的原始文件，输入有三项：

1. **`--author_id`**：连续索引 0~1019，对应 `style_features/{id}.pt`
2. **文本**：从该作者的测试页 `.wptt` 提取，或 `--text` 手动指定
3. **先验数据**：`datas_full.npy`、`mydict.npy`、`char_boxes.npy`、`char_lengths.npy`、`page_stats.npy`

生成过程：

- 用 `BoxLstm` 结合 `char_boxes.npy` 预测每个字的边界框
- 用 `DDPM` **批处理**逐行生成轨迹（一行所有字一次算完），`char_len` 从 `datas_full.npy` 查每个字的真实长度
- 用 `put_char` 把每个字缩放进框内，按 `page_stats.npy` 的行距、缩进、倾斜拼装为整页

## 踩坑记录

### 坑 1：`char_len` 用全局中位数导致复杂字崩坏

* **现象**：简单字（"今"、"我"）清晰，复杂字（"藏"、"散"）崩成乱线
* **根因**：训练时每个字 label 长度是它的**真实长度**（14~104 步），推理时统一用全局中位数 57 步
* **修复**：从 `datas_full.npy` 查每个字的真实长度

### 坑 2：`num_samples=3` 模糊风格特征

* **现象**：缓存 `.pt` 后生成质量比现场提取差
* **根因**：官方 `get_style_batch_10` **只采样 1 次**，多次平均把风格特征模糊化
* **修复**：改为 `num_samples=1`

### 坑 3：`L` 动态化导致位置编码异常

* **现象**：把 L 从 120 改小到 72 或 104 后整页崩溃
* **根因**：模型训练时的时间步长在 100~120 附近，动态改小后位置编码落在分布外
* **修复**：`L` 固定为 120

### 坑 4：原始 `datas.npy` 中 300 位作者数据不完整

* **现象**：`datas.npy` 有 1020 个 slot，其中 300 个只有约 260 字
* **根因**：HuggingFace 预处理版本中 DB1.2 的 300 位作者数据被截断
* **修复**：从 `.pot` 原始文件重建完整的 `datas_full.npy`，同时生成 `writer_id_map.json` 保存连续索引与原编号的对应

### 坑 5：逐字推理 GPU 虚忙

* **现象**：任务管理器显示 GPU 利用率 90%+，但功耗仅 50W
* **根因**：每次推理 `bs=1`，GPU 时间被 kernel launch 占据，算力利用率极低
* **修复**：改为一行内批处理（`bs = 该行字数`），单页耗时从数小时降到 400 秒

### 坑 6：U-Net 下采样长度必须对齐

* **现象**：`RuntimeError: The size of tensor a (26) must match the size of tensor b (25)`
* **根因**：1D U-Net 三层下采样（stride=2）要求输入长度是 8 的倍数
* **修复**：`L = 120`（8 × 15），永远对齐

## 参考文献

```bibtex
@inproceedings{ren2025decoupling,
  title={Decoupling Layout from Glyph in Online Chinese Handwriting Generation},
  author={Minsi Ren and Yan-Ming Zhang and Yi Chen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=DhHIw9Nbl1}
}
```

## 致谢

* **原论文**：Ren et al., ICLR 2025
* **原代码**：[singularityrms/OLHWG](https://github.com/singularityrms/OLHWG)
* **数据集**：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。参考：Cheng-Lin Liu et al., *CASIA Online and Offline Chinese Handwriting Databases*, ICDAR 2011.
* **预处理数据**：[HuggingFace: Immortalman12/OLHWD](https://huggingface.co/datasets/Immortalman12/OLHWD)

## 反馈与建议

如果在复现过程中遇到问题，欢迎通过 GitHub Issues 联系。

## 许可证

本项目采用 MIT 许可证。

**注意**：原论文、原代码及 CASIA 数据集的版权归各自作者/机构所有，使用本项目时请遵守相关学术使用限制，并引用原论文。