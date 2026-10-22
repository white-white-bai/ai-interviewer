"""M6：代码沙箱——让 Agent 能"执行真实代码"，但把风险关进笼子

Agent 的行动能力不止"调 LLM / 查数据库"，还可以执行代码获得真实反馈。
但候选人（或 LLM）提交的代码是不可信的：死循环、读写文件、联网……
所以绝不能在主进程里直接 exec——必须隔离子进程 + 强制超时。

隔离层次（教学版 → 生产版）：
- 本实现：独立子进程 + 超时强杀 + 临时目录隔离 + 全量捕获输出
- 生产版（M7 安全衔接）：Docker 容器 / 独立低权限账号 / 网络隔离 /
  内存与 CPU 限额（Windows 无 Unix rlimits，容器是标准解法）
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# 单次执行的默认超时（秒）：够跑正常练习代码，又能快速杀死死循环
DEFAULT_TIMEOUT = 5.0


def run_python(code: str, timeout_seconds: float = DEFAULT_TIMEOUT) -> dict:
    """在隔离子进程中执行一段 Python 代码，返回结构化结果（永不抛异常）

    隔离手段：
    1. 代码写入系统临时目录下的独立文件，工作目录也设在那里（不污染项目）
    2. subprocess.run 起独立进程执行，与主进程零共享状态
    3. timeout 到点由操作系统层面终止进程——死循环也伤不到主服务
    4. 无论发生什么（超时/崩溃/未知异常），都收敛成 dict 返回，
       让调用方（图节点）拿到确定性结果——降级兜底的雏形（M11 系统化）

    Args:
        code: 要执行的 Python 源代码文本
        timeout_seconds: 执行超时秒数，超时进程被强制终止
    """
    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp:
        script = Path(tmp) / "candidate_code.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmp,
            )
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            # 超时：进程已被强杀，尽力回收已产生的部分输出
            out = exc.stdout or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            return {
                "ok": False,
                "exit_code": None,
                "stdout": out,
                "stderr": f"执行超过 {timeout_seconds} 秒，进程被强制终止",
                "timed_out": True,
            }
        except Exception as exc:  # 沙箱自身的意外也要兜住
            return {
                "ok": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"沙箱内部错误：{exc}",
                "timed_out": False,
            }


def format_result(result: dict, max_chars: int = 600) -> str:
    """把沙箱结果整理成紧凑文本（写进 state、注入评估提示词用），超长截断

    Args:
        result: run_python 的返回值
        max_chars: stdout/stderr 各自的最大保留字符数
    """
    if result["ok"]:
        status = "成功"
    elif result["timed_out"]:
        status = "超时被强制终止"
    else:
        status = f"异常退出（exit_code={result['exit_code']}）"
    parts = [f"执行状态：{status}"]
    if result["stdout"].strip():
        parts.append(f"标准输出：\n{result['stdout'].strip()[:max_chars]}")
    if result["stderr"].strip():
        parts.append(f"标准错误：\n{result['stderr'].strip()[:max_chars]}")
    return "\n".join(parts)
