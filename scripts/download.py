# -*- coding: utf-8 -*-
"""从魔搭下载 int4 权重到本地缓存。技能自带权重时不需要跑这个。

用法
  python download.py            # 下载 int4（约 63MB）
  python download.py --check    # 只检查本地有没有，并校验 sha256
  python download.py --dir D:\\models   # 指定落点（之后设 XIAOYU_WEIGHTS 指过去）

说明
  - 下载后一定校验 sha256，不符即报错，不会静默通过
  - 网络失败重试 3 次（1s / 3s / 9s 退避）；仍失败给出人话提示，不抛裸 traceback
  - 若报 404 / NotFound，通常是模型仓库暂不可用（未发布或迁移），不是你的操作问题
"""
import argparse
import os
import shutil
import sys
import time

MODEL_ID = "fuyi163/xiaoyu-chinese-ai-detect"
FNAME = "int4_packed.pt"
RELPATH = "quant/" + FNAME
SHA256 = "80a2505fbc922ab33cc4e9875dd87b626225daba83bf980316c1ad07b045f2d9"

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SKILL, "assets")
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "xiaoyu-ai-detect")

MANUAL_HINT = (
    "手动下载：打开 https://modelscope.cn/models/%s\n"
    "          下 %s，放到 %s（或任意目录后设环境变量 XIAOYU_WEIGHTS 指向该目录）"
    % (MODEL_ID, RELPATH, DEFAULT_DIR)
)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_one(p):
    """返回 (是否存在, 校验结论)。校验结论：OK / MISMATCH:<实测> / 读不到。"""
    if not os.path.isfile(p):
        return False, None
    try:
        got = sha256(p)
    except OSError as e:
        return True, "读不到（%s）" % str(e)[:60]
    return True, ("OK" if got == SHA256 else "MISMATCH:%s" % got)


def do_check(dst, builtin):
    print("本地权重状态（期望 sha256 %s…）" % SHA256[:16])
    any_good = False
    for p, tag in ((builtin, "技能内置"), (dst, "用户缓存")):
        exists, verdict = check_one(p)
        if not exists:
            print("  %-8s 不存在  %s" % (tag, p))
            continue
        ok = (verdict == "OK")
        any_good = any_good or ok
        print("  %-8s %s  %.1f MB  sha256 %s"
              % (tag, p, os.path.getsize(p) / 1048576.0,
                 "校验通过" if ok else verdict))
    if not any_good:
        print("\n没有可用的权重。跑一次：python download.py")
        print(MANUAL_HINT)


def main():
    ap = argparse.ArgumentParser(description="下载晓语中文 AI 检测的 int4 权重")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="落点目录")
    ap.add_argument("--check", action="store_true", help="只查本地状态（含 sha256 校验）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    a = ap.parse_args()

    dst = os.path.join(a.dir, FNAME)
    builtin = os.path.join(ASSETS, FNAME)

    if a.check:
        do_check(dst, builtin)
        return

    if os.path.isfile(dst) and not a.force:
        exists, verdict = check_one(dst)
        if verdict == "OK":
            print("已存在且校验通过：%s（%.1f MB）"
                  % (dst, os.path.getsize(dst) / 1048576.0))
            return
        print("已存在但%s：%s" % (verdict or "无法校验", dst))
        print("重下请加 --force（会自动覆盖）")

    try:
        from modelscope import snapshot_download
    except ImportError:
        sys.exit("缺 modelscope：pip install modelscope\n" + MANUAL_HINT)

    os.makedirs(a.dir, exist_ok=True)
    print("从 %s 拉取 %s …" % (MODEL_ID, RELPATH))

    d, last = None, None
    for attempt in range(3):
        try:
            d = snapshot_download(MODEL_ID, allow_patterns=[RELPATH])
            break
        except KeyboardInterrupt:
            sys.exit("已中断")
        except Exception as e:
            last = e
            msg = str(e)
            print("  第 %d 次失败：%s" % (attempt + 1, msg[:160]))
            if attempt < 2:
                wait = 3 ** attempt
                print("  %d 秒后重试…" % wait)
                time.sleep(wait)

    if d is None:
        msg = str(last)
        low = msg.lower()
        print("\n下载失败，三次都没成功。")
        if any(k in low for k in ("404", "notfound", "not found", "notexist", "does not exist")):
            print("远端返回「找不到」——通常是模型仓库暂不可用（尚未发布或正在迁移），")
            print("不是你的操作问题。稍后重试，或按下面方式手动下载：")
        elif any(k in low for k in ("proxy", "timeout", "timed out", "connection", "ssl")):
            print("看着像网络/代理问题。若环境里设了 HTTP(S)_PROXY，试试先清掉再跑：")
            print("  Linux/macOS:  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy")
            print("  Windows CMD:  set HTTP_PROXY=  &  set HTTPS_PROXY=")
        else:
            print("原因见上面的报错。手动下载方式：")
        print(MANUAL_HINT)
        sys.exit(1)

    src = os.path.join(d, *RELPATH.split("/"))
    if not os.path.isfile(src):
        sys.exit("下载完成但没找到 %s（目录 %s）\n%s" % (RELPATH, d, MANUAL_HINT))

    shutil.copyfile(src, dst)
    got = sha256(dst)
    if got != SHA256:
        sys.exit("sha256 不符，文件已损坏，已保留在原处供排查：\n"
                 "  期望 %s\n  实测 %s\n删掉后重跑本脚本" % (SHA256, got))
    print("完成 → %s（%.1f MB，sha256 校验通过）"
          % (dst, os.path.getsize(dst) / 1048576.0))


if __name__ == "__main__":
    main()
