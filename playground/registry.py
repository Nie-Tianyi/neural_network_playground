"""模型注册表 + 热重载 —— 让你在 PyCharm 里改完模型，浏览器不用重启。

## 热重载为什么不能用 importlib.reload

`importlib.reload` 会**在原模块对象的命名空间里就地重新执行代码**。
旧版本定义的类仍然被别处引用着，于是你会看到"改了代码但注册表里还是老模型"
（甚至同一份代码注册出两个同名模型）。这是热重载最经典的坑。

这里的做法是：每次都用 `module_from_spec` + `exec_module` 建一个**全新的模块对象**，
从一个干净的命名空间里重新收集模型。旧模块连同它的类一起被丢弃。

## 注册方式：打标记，不用全局状态

不改用"注册到某个全局单例"，而是让装饰器往类上挂一个 `__playground_model__` 属性。
重载后扫描新模块的命名空间，把所有带标记的对象收上来。

这样没有全局可变状态，也就不存在"重载两次注册表里有两份"的问题。
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

MARKER = "__playground_model__"


@dataclass
class ModelSpec:
    """一个可被前端选中的模型。"""

    key: str
    title: str
    cls: type[nn.Module]
    description: str = ""
    problem: str | None = None  # None 表示分类/回归都能用
    configurable_shape: bool = False  # 是否响应前端的"加层/加神经元"按钮
    defaults: dict[str, Any] = field(default_factory=dict)  # 建议的超参覆盖

    def build(self, in_dim: int, out_dim: int, cfg) -> nn.Module:
        return self.cls(in_dim, out_dim, cfg)


def register(
    key: str,
    title: str | None = None,
    description: str = "",
    problem: str | None = None,
    configurable_shape: bool = False,
    defaults: dict[str, Any] | None = None,
):
    """把一个 nn.Module 注册成前端可选模型。

    用法::

        @register("mlp", title="MLP", configurable_shape=True)
        class MLP(nn.Module):
            def __init__(self, in_dim, out_dim, cfg):
                ...

    构造函数固定收这三个参数：
        in_dim  —— 输入特征数（等于当前勾选的特征个数）
        out_dim —— 输出维度，分类/回归都是 1
        cfg     —— Config 对象，模型从中读取激活函数、正则等
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        setattr(
            cls,
            MARKER,
            ModelSpec(
                key=key,
                title=title or cls.__name__,
                cls=cls,
                description=description or (cls.__doc__ or "").strip(),
                problem=problem,
                configurable_shape=configurable_shape,
                defaults=defaults or {},
            ),
        )
        return cls

    return decorator


class ModelRegistry:
    """扫描若干个 .py 文件，收集注册的模型；文件改动时自动重载。"""

    def __init__(self, files: list[Path]) -> None:
        self.files = [Path(f) for f in files]
        self._specs: dict[str, ModelSpec] = {}
        self._errors: dict[str, str] = {}
        self._mtimes: dict[Path, float] = {}
        # 首次加载直接做，调用方不用先手动 reload 一次。
        self.reload(force=True)

    # --- 查询 ---------------------------------------------------------------

    @property
    def specs(self) -> dict[str, ModelSpec]:
        return self._specs

    @property
    def errors(self) -> dict[str, str]:
        """文件名 → 错误信息。用户模型写挂了不影响内置模型可用。"""
        return self._errors

    def get(self, key: str) -> ModelSpec | None:
        return self._specs.get(key)

    def reload_if_stale(self) -> bool:
        """文件有改动就重载。返回是否真的重载了。"""
        changed = any(
            self._mtime(path) != self._mtimes.get(path) for path in self.files
        )
        if changed:
            self.reload()
        return changed

    # --- 加载 ---------------------------------------------------------------

    def reload(self, force: bool = False) -> None:
        specs: dict[str, ModelSpec] = {}
        errors: dict[str, str] = {}

        for path in self.files:
            if not path.exists():
                continue
            if not force and self._mtime(path) == self._mtimes.get(path):
                # 文件没变，但结果要重新汇总（可能另一个文件变了）
                specs.update(
                    {k: v for k, v in self._specs.items() if _from_file(v, path)}
                )
                continue
            try:
                module = _exec_fresh(path)
            except Exception as exc:  # noqa: BLE001 —— 见下
                # 这里执行的是**用户的代码**，抛什么都有可能：SyntaxError、
                # ImportError、甚至 torch 内部的 RuntimeError。一行行去枚举
                # 只会让"编辑到一半保存"时页面直接空掉。
                #
                # 所以宽泛捕获是刻意的：记下错误，但保留这个文件上一次成功的
                # 模型，让页面继续可用。这是 builtin_models.py 和 models.py
                # 分成两个文件的原因的延伸。
                errors[path.name] = f"{type(exc).__name__}: {exc}"
                specs.update(
                    {k: v for k, v in self._specs.items() if _from_file(v, path)}
                )
                self._mtimes[path] = self._mtime(path)
                continue

            for value in vars(module).values():
                spec = getattr(value, MARKER, None)
                if isinstance(spec, ModelSpec):
                    spec.defaults = {**spec.defaults, "_file": path.name}
                    specs[spec.key] = spec
            self._mtimes[path] = self._mtime(path)

        self._specs = specs
        self._errors = errors

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0


def _from_file(spec: ModelSpec, path: Path) -> bool:
    return spec.defaults.get("_file") == path.name


def _exec_fresh(path: Path):
    """在一个全新的模块对象里执行文件。

    和 importlib.reload 的区别：reload 是就地重置已有模块，
    这里每次都是一个新对象，旧模块的类不会被复用。

    模块名必须是 `playground` 的**直接子模块**（不能再多一层），
    否则文件里的 `from .config import ...` 会解析到不存在的包上。
    """
    module_name = f"playground._loaded_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    # 先放进 sys.modules：@dataclass 之类会去 sys.modules 里按 __module__ 找自己。
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def default_registry() -> ModelRegistry:
    """内置模型 + 用户模型。

    分成两个文件是有意的：内置的那份永远可用，你在 models.py 里写到一半
    保存（哪怕有语法错误），也还有模型可以跑。
    """
    package_dir = Path(__file__).parent
    return ModelRegistry(
        [
            package_dir / "builtin_models.py",
            package_dir / "models.py",
        ]
    )
