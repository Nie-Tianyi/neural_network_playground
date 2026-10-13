"""模型注册表与热重载。

热重载是这个项目的头号卖点（"在 IDE 里写完保存，浏览器里立刻能选到"），
所以值得单独测。用临时文件测，不去动用户真正在编辑的 models.py。
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest
import torch.nn as nn

from playground.registry import ModelRegistry

MODEL_TEMPLATE = '''
import torch.nn as nn
from playground.registry import register


@register({key!r}, title={title!r})
class {cls}(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        self.fc = nn.Linear(in_dim, {width})
        self.out = nn.Linear({width}, out_dim)

    def forward(self, x):
        return self.out(self.fc(x))
'''


def write_model(path: Path, cls: str, key: str, title: str, width: int = 4) -> None:
    path.write_text(
        MODEL_TEMPLATE.format(key=key, title=title, cls=cls, width=width),
        encoding="utf-8",
    )
    # 文件系统的时间戳精度有限（Windows 上尤其），连续写两次可能拿到同一个
    # mtime，于是"改动检测"会漏掉第二次。测试里显式把时间往后推。
    future = time.time() + 1
    os.utime(path, (future, future))


def test_loads_models_from_file(tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    write_model(path, "NetA", "a", "网络 A")
    registry = ModelRegistry([path])
    assert list(registry.specs) == ["a"]
    assert registry.specs["a"].title == "网络 A"
    assert registry.errors == {}


def test_hot_reload_picks_up_new_model(tmp_path: Path) -> None:
    """加一个新模型，重载后要能选到。"""
    path = tmp_path / "m.py"
    write_model(path, "NetA", "a", "网络 A")
    registry = ModelRegistry([path])

    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\n" + MODEL_TEMPLATE.format(key="b", title="网络 B", cls="NetB", width=6),
        encoding="utf-8",
    )
    future = time.time() + 2
    os.utime(path, (future, future))

    assert registry.reload_if_stale()
    assert set(registry.specs) == {"a", "b"}


def test_hot_reload_does_not_duplicate(tmp_path: Path) -> None:
    """反复重载不能把同一个模型注册两遍。

    这正是 importlib.reload 的坑：它就地重置已有模块，旧类还挂在别处引用上，
    于是同一个 key 会以两份不同的类存在。这里每轮都用全新的模块对象，
    所以必须干干净净地只有一份。
    """
    path = tmp_path / "m.py"
    write_model(path, "NetA", "a", "网络 A")
    registry = ModelRegistry([path])

    for _ in range(5):
        registry.reload(force=True)

    assert list(registry.specs) == ["a"]
    # 而且必须是**当前文件里**的那个类，不是第一版的残骸
    assert registry.specs["a"].cls.__name__ == "NetA"
    assert registry.specs["a"].cls.__module__.startswith("playground._loaded_")


def test_syntax_error_keeps_previous_models(tmp_path: Path) -> None:
    """模型文件写到一半保存（语法错误）时，页面不能空掉。

    这是把内置模型和用户模型分成两个文件的理由：编辑到一半时至少还有东西能跑。
    """
    path = tmp_path / "m.py"
    write_model(path, "NetA", "a", "网络 A")
    registry = ModelRegistry([path])

    path.write_text("def broken(:\n", encoding="utf-8")  # 语法错误
    future = time.time() + 2
    os.utime(path, (future, future))
    registry.reload_if_stale()

    # 上一个能用的版本仍然在，同时错误被记下来给前端显示
    assert list(registry.specs) == ["a"]
    assert "m.py" in registry.errors
    assert "SyntaxError" in registry.errors["m.py"]


def test_broken_file_does_not_hide_other_files(tmp_path: Path) -> None:
    """一个文件坏了，另一个文件里的模型要照常可用。"""
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    write_model(good, "GoodNet", "good", "好的")
    bad.write_text("import nonexistent_module_xyz\n", encoding="utf-8")

    registry = ModelRegistry([good, bad])
    assert list(registry.specs) == ["good"]
    assert "bad.py" in registry.errors


def test_fixing_a_broken_file_recovers(tmp_path: Path) -> None:
    """把语法错误改好之后，模型要能恢复出现（而不是要重启服务）。"""
    path = tmp_path / "m.py"
    path.write_text("def broken(:\n", encoding="utf-8")
    registry = ModelRegistry([path])
    assert registry.specs == {}
    assert registry.errors

    write_model(path, "NetA", "a", "网络 A")
    registry.reload_if_stale()

    assert list(registry.specs) == ["a"]
    assert registry.errors == {}


def test_builtin_models_survive_a_broken_user_file(tmp_path: Path) -> None:
    """真实场景：内置模型文件正常，用户文件挂了 —— 内置的必须还在。"""
    builtin = Path(__file__).parent.parent / "playground" / "builtin_models.py"
    user = tmp_path / "models.py"
    user.write_text("this is not python(\n", encoding="utf-8")

    registry = ModelRegistry([builtin, user])
    assert "mlp" in registry.specs
    assert "models.py" in registry.errors


def test_default_registry_has_builtin_mlp() -> None:
    from playground.registry import default_registry

    registry = default_registry()
    assert "mlp" in registry.specs
    assert registry.specs["mlp"].configurable_shape


def test_registered_model_metadata(tmp_path: Path) -> None:
    """注册时给的元信息要原样传到前端。"""
    path = tmp_path / "m.py"
    path.write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            from playground.registry import register


            @register("kan-ish", title="KAN 风格", description="说明文字",
                      problem="classification", configurable_shape=False,
                      defaults={"learningRate": 0.001})
            class Net(nn.Module):
                def __init__(self, in_dim, out_dim, cfg):
                    super().__init__()
                    self.fc = nn.Linear(in_dim, out_dim)

                def forward(self, x):
                    return self.fc(x)
            """
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry([path])
    spec = registry.specs["kan-ish"]
    assert spec.title == "KAN 风格"
    assert spec.description == "说明文字"
    assert spec.problem == "classification"
    assert spec.configurable_shape is False
    assert spec.defaults["learningRate"] == 0.001


def test_model_is_buildable(tmp_path: Path) -> None:
    """注册表里的模型要真的能被构造出来（构造函数签名对得上）。"""
    from playground.config import Config

    path = tmp_path / "m.py"
    write_model(path, "NetA", "a", "网络 A", width=7)
    registry = ModelRegistry([path])

    model = registry.specs["a"].build(3, 1, Config())
    assert isinstance(model, nn.Module)
    out = model(__import__("torch").zeros(5, 3))
    assert tuple(out.shape) == (5, 1)


def test_missing_file_is_ignored(tmp_path: Path) -> None:
    """文件不存在不该炸 —— 用户可能还没建 models.py。"""
    registry = ModelRegistry([tmp_path / "nope.py"])
    assert registry.specs == {}
    assert registry.errors == {}


@pytest.mark.parametrize("count", [1, 3])
def test_reload_keeps_all_models(tmp_path: Path, count: int) -> None:
    path = tmp_path / "m.py"
    source = "\n\n".join(
        MODEL_TEMPLATE.format(key=f"k{i}", title=f"网络 {i}", cls=f"Net{i}", width=4)
        for i in range(count)
    )
    path.write_text(source, encoding="utf-8")
    registry = ModelRegistry([path])
    assert len(registry.specs) == count

    registry.reload(force=True)
    assert len(registry.specs) == count
