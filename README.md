# OLHWG-PageSynth

## 项目简介

本项目是对 ICLR 2025 论文 **《Decoupling Layout from Glyph in Online Chinese Handwriting Generation》** 的复现与工程化改造。

论文提出分层生成框架：布局生成器预测每个字的边界框，字符合成器在框内生成对应风格的轨迹。本项目在此基础上：

1. 跑通论文的完整推理流水线（单字生成 → 行布局 → 整页拼装）
2. 将推理入口改为**文本驱动**：给定作者 ID 和一段文本，直接输出整页手写图
3. 定位并修复了推理阶段导致生成质量崩溃的三个关键缺陷

## 核心结果

模块 | 结果  
---|---  
单字生成 | 常用字可辨认，复杂字有改善空间  
行布局生成 | 字位置、字宽、行倾斜度与参考作者一致  
整页生成 | 多行页面完整输出  
风格模仿 | 生成样本风格与参考作者一致  

## 数据集

使用 CASIA-OLHWDB 的两个子集：

| 用途 | 数据集 | 文件 |
|---|---|---|
| 单字生成 | CASIA-OLHWDB **1.0-1.2**（单字） | `datas.npy` / `test_datas.npy` / `mydict.npy` |
| 行布局 / 整页生成 | CASIA-OLHWDB **2.0-2.2**（文本行） | `all_datas.npy` / `dictionary.npy` / `metadata.csv` |

论文确认两批数据由**同一批 1020 位书写者**产生。本项目利用这一点：从 1.x 的单字数据里提取作者风格，同时用 2.x 的行数据训练/验证布局生成器。

经过 `metadata.csv` 与 `datas.npy` 的索引匹配，最终 739 位作者有完整可用数据（其余因数据集本身的编号缺失被跳过）。

## 项目结构

```
OLHWG-PageSynth/
├── configs/
│   ├── default.yaml
│   └── author_split.yaml           # 自动生成
├── data/
│   ├── metadata.csv
│   └── processed/
│       ├── datas.npy
│       ├── test_datas.npy
│       ├── all_datas.npy
│       ├── dictionary.npy
│       ├── mydict.npy
│       ├── char_boxes.npy          # 00_build_char_boxes.py 生成
│       ├── char_lengths.npy        # 00b_build_char_lengths.py 生成
│       └── style_features/         # 01_prepare_data.py 生成
├── checkpoints/pretrained/
│   ├── unetddpm450000.pth
│   ├── style_encoder450000.pth
│   └── boxlstm3000.pth
├── src/
│   ├── models/                     # 网络定义
│   ├── datasets/                   # 数据加载
│   ├── pipeline/                   # 生成流水线
│   ├── evaluation/                 # 评估指标
│   └── utils/                      # 绘图、轨迹处理
├── scripts/
│   ├── 00_build_char_boxes.py
│   ├── 00b_build_char_lengths.py
│   ├── 01_prepare_data.py
│   ├── 04_infer_single_page.py     # 主力推理脚本
│   └── 05_run_batch_1020.py
├── outputs/pages/
└── README.md
```

## 快速开始

### 1. 环境

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib tqdm pyyaml fastdtw scikit-learn
```

### 2. 数据与权重

从 [HuggingFace: Immortalman12/OLHWD](https://huggingface.co/datasets/Immortalman12/OLHWD) 下载 `.npy` 数据集与预训练权重，按上面的目录结构放置。

### 3. 预处理（一次性）

```bash
python scripts/00_build_char_boxes.py
python scripts/00b_build_char_lengths.py
python scripts/01_prepare_data.py --config configs/default.yaml
```

三条命令分别生成：
* `char_boxes.npy` — 每个字的宽高先验
* `char_lengths.npy` — 每个字的平均轨迹长度
* `style_features/*.pt` — 739 位作者的风格特征缓存

### 4. 单页生成

```bash
python scripts/04_infer_single_page.py --config configs/default.yaml --author_id 1
```

输出到 `outputs/pages/author_0001/generated_page.png`。

自定义文本：

```bash
python scripts/04_infer_single_page.py --config configs/default.yaml \
    --author_id 15 \
    --text "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
```

### 5. 批量生成

```bash
python scripts/05_run_batch_1020.py --config configs/default.yaml
```

已生成的作者会自动跳过，支持中断续跑。

## 推理流程说明

`04_infer_single_page.py` 不依赖 `.wptt` 原始文件，也不需要 `all_datas.npy`。它的三个输入是：

1. **命令行 `--text`**：要生成什么内容
2. **`--author_id`**：模仿哪位作者的风格（对应 `style_features/{id}.pt`）
3. **`datas.npy` / `mydict.npy` / `char_boxes.npy` / `char_lengths.npy`**：模型需要的字典与先验

生成过程：

* 用 `BoxLstm` 结合 `char_boxes.npy` 预测每个字的边界框
* 用 `DDPM` 逐字生成轨迹，`char_len` 从 `char_lengths.npy` 查表
* 用 `put_char` 把每个字缩放进框内，按 `y_offset` 拼装为整页

## 踩坑记录

### 坑 1：`char_len` 用全局中位数导致复杂字崩坏

* 现象：简单字清晰，复杂字崩成乱线
* 原因：模型训练时每个字的 label 长度是它的真实长度，推理时统一用全局中位数 57
* 修复：改为从 `datas[writer_id - 1]` 查每个字的真实长度

### 坑 2：`num_samples=3` 模糊风格特征

* 现象：缓存风格特征后生成质量比现场提取差
* 原因：官方 `get_style_batch_10` 只采样 1 次，多次平均把风格特征模糊化
* 修复：改为 `num_samples=1`

### 坑 3：`L` 动态化导致位置编码异常

* 现象：把 L 从 120 改小后整页崩溃
* 原因：模型训练时的时间步分布集中在 100~120
* 修复：`L` 固定为 120，不做对齐

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

* 原论文：Ren et al., ICLR 2025
* 原代码：[singularityrms/OLHWG](https://github.com/singularityrms/OLHWG)
* 数据集：CASIA-OLHWDB，中国科学院自动化研究所
* 预处理数据：[Immortalman12/OLHWD](https://huggingface.co/datasets/Immortalman12/OLHWD)

## 许可证

MIT
