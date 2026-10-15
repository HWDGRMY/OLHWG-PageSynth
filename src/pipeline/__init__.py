"""整页生成流水线。"""
from .style_extractor import extract_style_features
from .line_assembler import generate_line, put_char
from .page_generator import generate_page

__all__ = ["extract_style_features", "generate_line", "put_char", "generate_page"]