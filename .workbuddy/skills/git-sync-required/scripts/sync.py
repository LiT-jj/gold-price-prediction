#!/usr/bin/env python3
"""一键 Git 同步脚本：暂存 -> 提交 -> 推送 -> 校验。

用法:
    python sync.py "提交信息"
    python sync.py --no-push "提交信息"    # 仅本地提交
    python sync.py --check                # 仅校验工作区状态

退出码:
    0  成功
    1  参数或环境错误
    2  提交/推送失败
"""

import argparse
import subprocess
import sys
from pathlib import Path

# scripts/ -> git-sync-required/ -> skills/ -> .workbuddy/ -> 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[4]


def run(args, check=True, capture=True):
    """执行 git 命令并返回结果。"""
    result = subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if check and result.returncode != 0:
        sys.stdout.write(result.stdout or "")
        raise SystemExit(result.returncode)
    return result


def has_remote():
    result = run(["remote"], check=False)
    return "origin" in (result.stdout or "")


def main():
    parser = argparse.ArgumentParser(description="项目 Git 自动化同步")
    parser.add_argument("message", nargs="?", help="commit message")
    parser.add_argument("--no-push", action="store_true", help="跳过推送")
    parser.add_argument("--check", action="store_true", help="仅校验状态")
    args = parser.parse_args()

    if run(["rev-parse", "--is-inside-work-tree"], check=False).returncode != 0:
        print("[X] 当前目录不是 Git 仓库")
        return 1

    if args.check:
        return do_check()

    if not args.message:
        print("[X] 必须提供 commit message 参数")
        return 1

    # 1. 展示待提交改动
    print("=== 待提交改动 ===")
    run(["status", "--short"], check=False)
    print()

    # 2. 暂存所有已跟踪文件的改动 + 新增文件
    run(["add", "-u"])
    run(["add", "."])
    print("=== 暂存后 ===")
    run(["status", "--short"], check=False)
    print()

    # 3. 提交（无改动则跳过）
    staged = run(["diff", "--cached", "--quiet"], check=False).returncode
    if staged == 0:
        print("[=] 工作区无改动，无需提交")
    else:
        result = run(["commit", "-m", args.message], check=False)
        if result.returncode != 0:
            print("[X] 提交失败")
            return 2
        print("[OK] 提交完成")

    # 4. 推送
    if args.no_push:
        print("[=] 已跳过推送 (--no-push)")
    elif not has_remote():
        print("[!] 未配置 origin 远程仓库，仅完成本地提交")
        print("    请添加远程地址: git remote add origin <repo-url>")
    else:
        result = run(["push", "origin", "HEAD"], check=False)
        print(result.stdout or "")
        if result.returncode != 0:
            print("[X] 推送失败，请检查凭据或先执行 git pull --rebase")
            return 2
        print("[OK] 推送完成")

    print()
    return do_check()


def do_check():
    """校验工作区与远程一致性。"""
    print("=== 校验 ===")
    status = run(["status", "--short"], check=False).stdout.strip()
    if status:
        print("[X] 工作区仍存在未提交改动:")
        print(status)
        return 2
    print("[OK] 工作区干净")

    log = run(["log", "--oneline", "-1"], check=False).stdout.strip()
    print(f"    最新提交: {log}")

    if has_remote():
        local = run(["rev-parse", "HEAD"], check=False).stdout.strip()
        remote = run(["rev-parse", "origin/main"], check=False)
        if remote.returncode == 0:
            if local == remote.stdout.strip():
                print("[OK] 本地与 origin/main 已同步")
            else:
                print("[!] 本地与 origin/main 不一致，推送未生效")
                return 2
        else:
            print("[!] 无法读取 origin/main，推送可能未完成")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
