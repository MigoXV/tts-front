import json
from pathlib import Path
from typing import Dict


def load_voice_map(voice_map_path: Path) -> Dict[str, str]:
    """加载语音映射文件，返回语音名称到语音ID的映射字典。

    Args:
        voice_map_path (Path): 语音映射文件路径，支持JSON格式。

    Returns:
        Dict: 语音名称到语音ID的映射字典。
    """
    if not voice_map_path.exists():
        raise FileNotFoundError(f"语音映射文件未找到: {voice_map_path}")

    with voice_map_path.open("r", encoding="utf-8") as f:
        voice_map = json.load(f)

    return voice_map
