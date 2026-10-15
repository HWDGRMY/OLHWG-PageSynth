"""
CASIA-OLHWDB .wptt 文件解析模块（最终稳定版）
- 非标签字段：小端读取
- 标签字段：原始字节读取，兼容 GBK 大端/小端、ASCII 双向
"""

import struct
import numpy as np


def _read_uint16_le(f):
    """小端 uint16"""
    data = f.read(2)
    if len(data) < 2:
        raise EOFError("Unexpected end of file")
    return struct.unpack('<H', data)[0]

def _read_uint32_le(f):
    """小端 uint32"""
    data = f.read(4)
    if len(data) < 4:
        raise EOFError("Unexpected end of file")
    return struct.unpack('<I', data)[0]

def _read_string(f, length):
    data = f.read(length)
    if len(data) < length:
        raise EOFError("Unexpected end of file")
    return data.decode('ascii', errors='ignore').strip('\x00').strip()


def _decode_tag(raw):
    """
    解码 2 字节标签。
    依次尝试：
    1. GBK 大端字节序
    2. GBK 小端字节序
    3. 半角 ASCII（0x00 + ASCII 或 ASCII + 0x00）
    异常码 0xffff 返回空串。
    """
    if len(raw) < 2:
        return '?'
    if raw == b'\xff\xff' or raw == b'\x00\x00':
        return ''
    b0, b1 = raw[0], raw[1]

    # 1. GBK 大端
    try:
        char = raw.decode('gbk')
        if char.isprintable():
            return char
    except UnicodeDecodeError:
        pass

    # 2. GBK 小端（交换字节）
    swapped = bytes([b1, b0])
    try:
        char = swapped.decode('gbk')
        if char.isprintable():
            return char
    except UnicodeDecodeError:
        pass

    # 3. 半角 ASCII：0x00 在前或 ASCII 在前
    if b0 == 0x00 and chr(b1).isprintable():
        return chr(b1)
    if b1 == 0x00 and chr(b0).isprintable():
        return chr(b0)

    # 4. 全角符号或未知，返回占位符
    return '?'


def load_wptt(wptt_path):
    """
    解析 .wptt 文件。
    返回字典包含：
    - format_code, illustration, code_type, code_length,
      data_type, sample_length, page_index
    - strokes: list of np.ndarray (num_points, 2)
    - lines: list of dict ('stroke_indices', 'text')
    """
    result = {
        'format_code': '',
        'illustration': '',
        'code_type': '',
        'code_length': 0,
        'data_type': '',
        'sample_length': 0,
        'page_index': 0,
        'strokes': [],
        'lines': []
    }

    with open(wptt_path, 'rb') as f:
        # 文件头
        header_size = _read_uint32_le(f)
        format_code = _read_string(f, 8)
        result['format_code'] = format_code

        illustration_len = header_size - 54
        illustration = f.read(illustration_len).decode('ascii', errors='ignore').strip('\x00').strip()
        result['illustration'] = illustration

        code_type = _read_string(f, 20)
        result['code_type'] = code_type

        code_length = _read_uint16_le(f)
        result['code_length'] = code_length

        data_type = _read_string(f, 20)
        result['data_type'] = data_type

        sample_length = _read_uint32_le(f)
        result['sample_length'] = sample_length

        page_index = _read_uint32_le(f)
        result['page_index'] = page_index

        # 总笔画数
        stroke_num = _read_uint32_le(f)

        # 所有笔画
        strokes = []
        for _ in range(stroke_num):
            point_num = _read_uint16_le(f)
            points = []
            for _ in range(point_num):
                x = _read_uint16_le(f)
                y = _read_uint16_le(f)
                points.append([x * 0.1, y * 0.1])
            strokes.append(np.array(points, dtype=np.float32))
        result['strokes'] = strokes

        # 行数
        line_num = _read_uint16_le(f)

        # 每行信息
        for _ in range(line_num):
            line_stroke_num = _read_uint16_le(f)
            stroke_indices = [_read_uint16_le(f) for _ in range(line_stroke_num)]
            line_char_num = _read_uint16_le(f)

            # 标签：逐字符读取原始 2 字节并解码
            text = ''
            for _ in range(line_char_num):
                raw = f.read(2)
                if len(raw) < 2:
                    break
                text += _decode_tag(raw)

            result['lines'].append({
                'stroke_indices': stroke_indices,
                'text': text
            })

    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        data = load_wptt(path)
        print(f"Page {data['page_index']} loaded, strokes: {len(data['strokes'])}, lines: {len(data['lines'])}")
        for i, line in enumerate(data['lines']):
            print(f"Line {i+1}: {line['text']}")
    else:
        print("Usage: python load_wptt.py <path_to_wptt>")