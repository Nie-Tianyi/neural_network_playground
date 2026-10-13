"""入口：uv run main.py

起来之后浏览器会自动打开 http://127.0.0.1:8000
（不想自动开就加 --no-browser，想换端口加 --port 8080）
"""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser

import uvicorn

from playground.server import create_app


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_port(preferred: int) -> int:
    """端口被占就往后找 —— 上一次没关干净是很常见的情况，不该因此启动失败。"""
    for candidate in range(preferred, preferred + 20):
        if _port_is_free(candidate):
            return candidate
    raise SystemExit(f"端口 {preferred}-{preferred + 19} 都被占用了")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch Neural Network Playground")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true", help="改 Python 代码自动重启（开发用）")
    args = parser.parse_args()

    port = _pick_port(args.port)
    url = f"http://{args.host}:{port}"

    print()
    print("  PyTorch Playground")
    print(f"  -> {url}")
    print()
    print("  在 playground/models.py 里写你自己的网络，")
    print("  保存后页面上的模型下拉框会自动更新（服务运行中也可以）。")
    print("  按 Ctrl+C 退出。")
    print()

    if not args.no_browser:
        # 等 uvicorn 把端口 bind 起来再开浏览器，否则会打开一个错误页
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if args.reload:
        # reload 模式必须传 import 字符串（子进程里要能重新导入），
        # 此时 uvicorn 需要 factory=True 才会去调用 create_app()
        uvicorn.run(
            "playground.server:create_app",
            host=args.host,
            port=port,
            log_level="warning",
            reload=True,
            factory=True,
        )
    else:
        uvicorn.run(create_app(), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
