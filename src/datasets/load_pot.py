"""
CASIA-OLHWDB .pot 文件解析模块
- 顺序解析，严格校验结束标记
- GBK 标签正确解码
- 提供两种输出：图像 (load_pot_samples) 和轨迹 (load_pot_trajs)
"""

import struct
import numpy as np
import cv2


def _decode_gbk(tag_code_16):
    """解码 16 位 GBK 标签，兼容汉字、数字、字母、标点。"""
    if tag_code_16 in (0x0000, 0xffff):
        return '?'
    # 半角 ASCII（高字节为 0）
    if (tag_code_16 & 0xFF00) == 0:
        ascii_code = tag_code_16 & 0xFF
        char = chr(ascii_code)
        if char.isprintable():
            return char
    # 半角 ASCII（低字节为 0）
    if (tag_code_16 & 0x00FF) == 0:
        ascii_code = (tag_code_16 >> 8) & 0xFF
        char = chr(ascii_code)
        if char.isprintable():
            return char
    # GBK 大端
    try:
        char = struct.pack('>H', tag_code_16).decode('gbk')
        if char.isprintable():
            return char
    except UnicodeDecodeError:
        pass
    # GBK 小端
    try:
        char = struct.pack('<H', tag_code_16).decode('gbk')
        if char.isprintable():
            return char
    except UnicodeDecodeError:
        pass
    return '?'


def _draw_strokes(strokes, padding=5):
    """根据笔画坐标生成白底黑字画布。"""
    all_points = []
    for stroke in strokes:
        all_points.extend(stroke)
    if not all_points:
        return np.full((32, 32), 255, dtype=np.uint8)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin + 2 * padding
    height = ymax - ymin + 2 * padding
    canvas = np.full((height, width), 255, dtype=np.uint8)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        for i in range(1, len(stroke)):
            pt1 = (stroke[i - 1][0] - xmin + padding, stroke[i - 1][1] - ymin + padding)
            pt2 = (stroke[i][0] - xmin + padding, stroke[i][1] - ymin + padding)
            cv2.line(canvas, pt1, pt2, 0, thickness=2)
    return canvas


def _read_pot_once(f):
    """从已打开的文件句柄里读一个样本。

    Returns:
        (label, strokes) 或 None（文件结束）
    """
    header = f.read(8)
    if len(header) < 8:
        return None
    try:
        _, tag_code_raw, stroke_num = struct.unpack('<HIH', header)
    except struct.error:
        return None

    tag_code = tag_code_raw & 0xFFFF
    label = _decode_gbk(tag_code)

    strokes = []
    for _ in range(stroke_num):
        stroke = []
        while True:
            b = f.read(4)
            if len(b) < 4:
                return None
            x, y = struct.unpack('<hh', b)
            if x == -1 and y == 0:
                break
            stroke.append((x, y))
        strokes.append(stroke)

    end_bytes = f.read(4)
    if len(end_bytes) < 4:
        return None
    ex, ey = struct.unpack('<hh', end_bytes)
    if not (ex == -1 and ey == -1):
        return None

    return label, strokes


def load_pot_samples(pot_path, normalize_size=256, max_samples=None):
    """读取 .pot 文件，返回所有字符样本的 (image, label) 列表。"""
    samples = []
    with open(pot_path, 'rb') as f:
        while True:
            if max_samples is not None and len(samples) >= max_samples:
                break
            result = _read_pot_once(f)
            if result is None:
                break
            label, strokes = result

            canvas = _draw_strokes(strokes)
            if normalize_size is not None:
                canvas = cv2.resize(canvas, (normalize_size, normalize_size),
                                    interpolation=cv2.INTER_AREA)
                canvas = np.where(canvas < 128, 0, 255).astype(np.uint8)

            samples.append((canvas, label))
    return samples


def load_pot_trajs(pot_path):
    """读取 .pot 文件，返回 [(char, traj), ...]。

    traj 形状 (N, 3)，格式与 datas.npy 兼容：
        - 前两列：相对坐标（除以字高归一化 + 转相对位移）
        - 第三列：state（1=落笔，-1=抬笔）
    """
    samples = []
    with open(pot_path, 'rb') as f:
        while True:
            result = _read_pot_once(f)
            if result is None:
                break
            label, strokes = result

            pts = []
            for s in strokes:
                for p in s:
                    pts.append([float(p[0]), float(p[1]), 1.0])
                if pts:
                    pts[-1][2] = -1.0

            if not pts:
                continue
            traj = np.array(pts, dtype=np.float32)

            # 归一化：中心平移到原点，高度缩放到 1
            x, y = traj[:, 0], traj[:, 1]
            x_c = (x.max() + x.min()) / 2.0
            y_c = (y.max() + y.min()) / 2.0
            h = max(float(y.max() - y.min()), 1e-3)
            traj[:, 0] = (traj[:, 0] - x_c) / h
            traj[:, 1] = (traj[:, 1] - y_c) / h

            # 绝对 → 相对（一阶差分）
            delta = traj.copy()
            delta[1:, 0] = traj[1:, 0] - traj[:-1, 0]
            delta[1:, 1] = traj[1:, 1] - traj[:-1, 1]

            if label and label != '?':
                samples.append((label, delta))

    return samples


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        samples = load_pot_samples(path)
        print(f"Loaded {len(samples)} samples from {path}")
        if samples:
            img, label = samples[0]
            print(f"First sample label: {label}, shape: {img.shape}")
            cv2.imwrite('sample_pot.png', img)
    else:
        print("Usage: python load_pot.py <path_to_pot>")