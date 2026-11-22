#!/usr/bin/env python3
"""Research project template tool: init a project, open an experiment card, check it, audit and prune artifacts.

Standard library only. Git is called with cwd instead of `git -C` so old git versions work.
"""
import argparse
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
TEMPLATE = HERE.parent.parent / "assets" / "template"
PLACEHOLDER = "【待填"
TYPES = ("trunk", "ablation", "analysis", "spike")
LINE_LIMITS = {
    "AGENTS.md": 80, "CLAUDE.md": 10, "STATE.md": 40, "README.md": 80,
    "docs/PROJECT.md": 150, "docs/DATA.md": 150, "docs/PIPELINE.md": 120, "docs/DECISIONS.md": 200,
}
CARD_LINES = 40
MAX_QUEUE = 5
MAX_FIGURES = 4
MAX_TRACKED_KB = 1024
ALLOWED_MD = set(LINE_LIMITS) | {"EXPERIMENTS.md"}
SKIP_MD_PREFIXES = ("scratch/", "runs/", "third_party/")
CODE_DIRS = ("src", "configs", "tests")
ROOT_DIRS = {"src", "configs", "experiments", "tests", "tools", "docs", "scratch", "runs", "data"}
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
KEPT_CHECKPOINTS = {"best.pt", "last.pt"}
MISPLACED_DIRS = re.compile(r"(?i)(cache|site-packages|^\.?venv$|^node_modules$)")
DUPLICATE_MIN_BYTES = 100 * 1024 ** 2
AUDIT_WARN_GB = 10
GB = 1024 ** 3
CARD = """# {id}

类型：{type}　状态：queued

## 问题
【待填：一句话，能用一个数或一张图回答】

## 影响的决定
- 【待填：结果 A】：下一步做什么。
- 【待填：结果 B】：下一步做什么。

## 预测
【待填：方向和大致幅度】

## 推翻条件
【待填：看到什么结果就承认预测错了】

## 相对主干的改动
【待填：只有一处，写 configs/exp/{id}.yaml 改了什么】

## 评测
同 PIPELINE.md。

## 计算和存储预算
【待填：GPU 小时或墙钟时间；存储上限 N GB（runs/ 下全部产物）；达到什么就停】

## 要保存的产物
result.json；checkpoints/best.pt；【待填：其余逐项写文件名，没写的不保存】

## 结果
【运行后填写：数字、与主干的差、结论、这个实验不能说明什么】
"""


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True)
    return result.returncode, result.stdout


def find_root(start):
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "STATE.md").is_file() and (candidate / "EXPERIMENTS.md").is_file():
            return candidate
    sys.exit("找不到项目根目录：需要同时有 STATE.md 和 EXPERIMENTS.md。新项目先运行 init。")


def tag_prefix(root):
    """Path of a sub-project inside a shared repository ("" at the top), used to namespace trunk tags."""
    code, out = git(root, "rev-parse", "--show-prefix")
    return out.strip() if code == 0 else ""


def lines_of(path):
    return path.read_text(encoding="utf-8").splitlines()


def cmd_init(args):
    if not TEMPLATE.is_dir():
        sys.exit("init 需要从 skill 的 scripts/project.py 运行，项目里的 tools/project.py 不带模板。")
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", args.name):
        sys.exit("--name 必须是合法的 Python 包名（小写字母、数字、下划线）。")
    root = Path(args.dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    today = args.date or datetime.date.today().isoformat()
    if git(root, "rev-parse", "--is-inside-work-tree")[0] != 0:
        git(root, "init")
        print("已运行 git init。")
    prefix = tag_prefix(root)
    created, skipped = [], []
    for source in sorted(TEMPLATE.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(TEMPLATE).as_posix().replace("__PKG__", args.name)
        if relative == "gitignore":
            relative = ".gitignore"
        target = root / relative
        if target.exists():
            skipped.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        for key, value in (("{{PKG}}", args.name), ("{{NAME}}", args.title or args.name), ("{{DATE}}", today),
                           ("{{TAG}}", prefix)):
            text = text.replace(key, value)
        target.write_text(text, encoding="utf-8")
        created.append(relative)
    tool = root / "tools" / "project.py"
    if tool.exists():
        skipped.append("tools/project.py")
    else:
        tool.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(HERE), str(tool))
        created.append("tools/project.py")
    for name in ("runs", "scratch"):
        (root / name).mkdir(exist_ok=True)
    if prefix:
        print("这是共享仓库里的子项目 {}：主干 tag 写成 {}trunk-v<N>，避免和其他子项目冲突。".format(prefix, prefix))
    print("创建 {} 个文件，跳过 {} 个已存在的文件。".format(len(created), len(skipped)))
    for relative in skipped:
        print("  跳过：" + relative)
    print("下一步：填写 docs/ 和 STATE.md 里的【待填】，然后提交。")


def cmd_new(args):
    root = find_root(args.root)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        sys.exit("slug 只能用小写字母、数字和连字符。")
    exp_id = "{}-{}".format(args.date or datetime.date.today().isoformat(), args.slug)
    directory = root / "experiments" / exp_id
    if directory.exists():
        sys.exit("已存在：experiments/" + exp_id)
    directory.mkdir(parents=True)
    (directory / "CARD.md").write_text(CARD.format(id=exp_id, type=args.type), encoding="utf-8")
    row = "| {id} | {type} | 【待填】 | 【待填】 | queued | 【待填】 | [卡片](experiments/{id}/CARD.md) |\n".format(
        id=exp_id, type=args.type)
    registry = root / "EXPERIMENTS.md"
    text = registry.read_text(encoding="utf-8")
    registry.write_text(text + ("" if text.endswith("\n") else "\n") + row, encoding="utf-8")
    print("已创建 experiments/{}/CARD.md，并在 EXPERIMENTS.md 追加一行。".format(exp_id))
    print("下一步：填写卡片，交给用户批准后提交 exp({}): card。".format(exp_id))


def registry_rows(root):
    rows = {}
    for line in lines_of(root / "EXPERIMENTS.md"):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and cells[0] not in ("id", "") and not set(cells[0]) <= set("-: "):
            rows[cells[0]] = cells
    return rows


def section(lines, title):
    body, inside = [], False
    for line in lines:
        if line.startswith("## "):
            inside = line[3:].strip().startswith(title)
            continue
        if inside:
            body.append(line)
    return body


def tree_size(path, seen=None):
    """Bytes under path without following symlinks; hard links are counted once."""
    seen = set() if seen is None else seen
    paths = [str(path)] if not path.is_dir() or path.is_symlink() else (
        os.path.join(base, name) for base, _, names in os.walk(str(path)) for name in names)
    total = 0
    for item in paths:
        try:
            stat = os.lstat(item)
        except OSError:
            continue
        if (stat.st_dev, stat.st_ino) not in seen:
            seen.add((stat.st_dev, stat.st_ino))
            total += stat.st_size
    return total


def card_info(root, exp_id):
    """Storage budget in GB and checkpoint file names the card asks to keep."""
    card = root / "experiments" / exp_id / "CARD.md"
    if not card.is_file():
        return None, set()
    lines = lines_of(card)
    budget = re.search(r"存储[^0-9\n]{0,10}(\d+(?:\.\d+)?)\s*G", "\n".join(section(lines, "计算和存储预算")))
    keep = set(re.findall(r"[\w.-]+(?:\.pt|\.pth|\.ckpt|\.safetensors)", "\n".join(section(lines, "要保存的产物"))))
    return (float(budget.group(1)) if budget else None), keep


def fingerprint(path, size):
    digest = hashlib.sha256(str(size).encode())
    with open(str(path), "rb") as handle:
        digest.update(handle.read(1024 ** 2))
        handle.seek(max(size - 1024 ** 2, 0))
        digest.update(handle.read(1024 ** 2))
    return digest.hexdigest()


def find_misplaced(directory, found):
    for base, dirs, _ in os.walk(str(directory)):
        for name in list(dirs):
            if MISPLACED_DIRS.search(name):
                path = Path(base) / name
                found.append((path, tree_size(path)))
                dirs.remove(name)


def collect(root, extra_runs=(), only=None, duplicates=True):
    """Find redundant artifacts. Nothing is deleted here."""
    rows = registry_rows(root)
    report = {key: [] for key in ("runs", "checkpoints", "misplaced", "over_budget", "unregistered", "duplicates")}
    big = {}
    for runs in [root / "runs"] + [Path(p).resolve() for p in extra_runs]:
        if not runs.is_dir():
            continue
        for run in sorted(p for p in runs.iterdir() if p.is_dir() and p.name != "_cache"):
            if only and run.name != only:
                continue
            size = tree_size(run)
            report["runs"].append((run, size))
            if run.name not in rows:
                report["unregistered"].append((run, size))
            status = rows.get(run.name, [""] * 7)[4]
            budget, keep = card_info(root, run.name)
            if budget is not None and size > budget * GB:
                report["over_budget"].append((run, size, budget))
            kept = KEPT_CHECKPOINTS - ({"last.pt"} if status in ("done", "stopped") else set()) | keep
            find_misplaced(run, report["misplaced"])
            for base, dirs, names in os.walk(str(run)):
                dirs[:] = [d for d in dirs if not MISPLACED_DIRS.search(d)]
                for name in names:
                    if name.endswith(CHECKPOINT_SUFFIXES) and name not in kept:
                        path = Path(base) / name
                        report["checkpoints"].append((path, path.lstat().st_size))
        if duplicates:
            for base, _, names in os.walk(str(runs)):
                for name in names:
                    path = os.path.join(base, name)
                    stat = os.lstat(path)
                    if stat.st_size >= DUPLICATE_MIN_BYTES and not os.path.islink(path):
                        big.setdefault(stat.st_size, {}).setdefault((stat.st_dev, stat.st_ino), Path(path))
    experiments = root / "experiments"
    if experiments.is_dir():
        for directory in sorted(p for p in experiments.iterdir() if p.is_dir()):
            if not only or directory.name == only:
                find_misplaced(directory, report["misplaced"])
    for size, by_inode in big.items():
        if len(by_inode) < 2:
            continue
        groups = {}
        for path in by_inode.values():
            groups.setdefault(fingerprint(path, size), []).append(path)
        for paths in groups.values():
            if len(paths) > 1 and (not only or any(only in p.parts for p in paths)):
                report["duplicates"].append((sorted(paths), size))
    return report


def show(root, path):
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def cmd_audit(args):
    root = find_root(args.root)
    report = collect(root, args.runs, args.exp)
    print("## 各 run 的大小")
    for run, size in report["runs"]:
        budget = card_info(root, run.name)[0]
        print("  {:7.2f}G  {}{}".format(size / GB, show(root, run), "（预算 {}G）".format(budget) if budget else ""))
    removable = report["checkpoints"] + report["misplaced"]
    for title, key in (("冗余 checkpoint：除 best.pt、last.pt 和卡片列出的文件之外", "checkpoints"),
                       ("放错位置的缓存和环境", "misplaced")):
        items = report[key]
        print("## {}（{} 项，{:.2f}G）".format(title, len(items), sum(s for _, s in items) / GB))
        for path, size in items:
            print("  {:7.2f}G  {}".format(size / GB, show(root, path)))
    for run, size, budget in report["over_budget"]:
        print("## 超出存储预算：{} 用了 {:.2f}G，预算 {}G".format(show(root, run), size / GB, budget))
    if report["unregistered"]:
        print("## 没有登记在 EXPERIMENTS.md 的 run 目录（只报告，不进清单）")
        for run, size in report["unregistered"]:
            print("  {:7.2f}G  {}".format(size / GB, show(root, run)))
    if report["duplicates"]:
        print("## 内容相同的大文件（只报告，不进清单；保留哪份由用户决定）")
        for paths, size in report["duplicates"]:
            print("  {:7.2f}G × {}：{}".format(size / GB, len(paths), "、".join(show(root, p) for p in paths)))
    scratch = root / "scratch"
    if scratch.is_dir():
        print("## scratch/ 共 {:.2f}G".format(tree_size(scratch) / GB))
    total = sum(size for _, size in removable)
    if not removable:
        print("可回收 0G，无需清理。")
        return 0
    scratch.mkdir(exist_ok=True)
    listing = scratch / "prune-{}{}.txt".format(datetime.date.today().isoformat(), "-" + args.exp if args.exp else "")
    listing.write_text("".join(str(path) + "\n" for path, _ in removable), encoding="utf-8")
    print("可回收 {:.2f}G，清单：{}".format(total / GB, show(root, listing)))
    print("下一步：把清单交给用户确认；确认后运行 tools/project.py prune {} --yes".format(show(root, listing)))
    return 0


def cmd_prune(args):
    root = find_root(args.root)
    bases = [p.resolve() for p in [root / "runs", root / "experiments", root / "scratch"] + [Path(r) for r in args.runs]]
    listing = Path(args.list)
    entries = [line.strip() for line in lines_of(listing) if line.strip() and not line.startswith("#")]
    targets, refused = [], []
    for entry in entries:
        path = Path(entry) if Path(entry).is_absolute() else root / entry
        parent = path.parent.resolve()
        target = parent / path.name
        inside = any(base == parent or base in parent.parents for base in bases)
        if not inside or path.name in ("CARD.md", "result.json"):
            refused.append(target)
        elif os.path.lexists(str(target)):
            targets.append(target)
        else:
            print("已不存在，跳过：" + show(root, target))
    for target in refused:
        print("拒绝：{} 不在 runs/、experiments/、scratch/ 或 --runs 之下，或者是卡片和结果文件。".format(target))
    if refused:
        print("清单里有被拒绝的路径，什么也没删。")
        return 1
    seen = set()
    sizes = [tree_size(target, seen) for target in targets]
    for target, size in zip(targets, sizes):
        print("  {:7.2f}G  {}".format(size / GB, show(root, target)))
    print("共 {} 项，{:.2f}G。".format(len(targets), sum(sizes) / GB))
    if not args.yes:
        print("没有删除任何东西。用户确认后加 --yes 执行。")
        return 0
    for target in targets:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(str(target))
        else:
            target.unlink()
    left = [t for t in targets if os.path.lexists(str(t))]
    for target in left:
        print("仍存在：" + show(root, target))
    print("已删除 {} 项，释放 {:.2f}G。".format(len(targets) - len(left), sum(sizes) / GB))
    return 1 if left else 0


def cmd_check(args):
    root = find_root(args.root)
    errors, warnings = [], []
    in_git = git(root, "rev-parse", "--is-inside-work-tree")[0] == 0
    if in_git:
        files = git(root, "ls-files")[1].splitlines()
        files += git(root, "ls-files", "--others", "--exclude-standard")[1].splitlines()
    else:
        errors.append("项目不在 git 仓库里。")
        files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    files = sorted({f for f in files if (root / f).is_file()})

    for relative, limit in LINE_LIMITS.items():
        path = root / relative
        if path.is_file() and len(lines_of(path)) > limit:
            errors.append("{} 有 {} 行，上限 {} 行：合并或删减。".format(relative, len(lines_of(path)), limit))
    for relative in files:
        if not relative.endswith(".md") or relative.startswith(SKIP_MD_PREFIXES):
            continue
        if re.fullmatch(r"experiments/[^/]+/CARD\.md", relative):
            if len(lines_of(root / relative)) > CARD_LINES:
                errors.append("{} 超过 {} 行。".format(relative, CARD_LINES))
        elif relative not in ALLOWED_MD:
            errors.append("多余的 Markdown 文件：{}。内容并入已有文件，或先问用户。".format(relative))

    if in_git:
        dirty = git(root, "status", "--porcelain", "--", *CODE_DIRS)[1].splitlines()
        if dirty:
            errors.append("{} 下有 {} 个未提交或未跟踪的文件，正式运行前先提交。例如：{}".format(
                "、".join(CODE_DIRS), len(dirty), dirty[0].strip()))
        for relative in git(root, "ls-files")[1].splitlines():
            path = root / relative
            if path.is_file() and path.stat().st_size > MAX_TRACKED_KB * 1024:
                errors.append("git 里有超过 {} KB 的文件：{}。".format(MAX_TRACKED_KB, relative))

    state = lines_of(root / "STATE.md")
    trunk = "\n".join(section(state, "当前主干"))
    match = re.search(r"commit[:：]\s*`?([0-9a-f]{7,40})\b", trunk)
    if match is None:
        warnings.append("STATE.md 还没有写主干的 commit：主干未锁定，此时只做 trunk 类型的实验。")
    elif in_git and git(root, "cat-file", "-e", match.group(1) + "^{commit}")[0] != 0:
        errors.append("STATE.md 里的主干 commit {} 不存在。".format(match.group(1)))
    tag = re.search(r"tag[:：]\s*`?([^\s`【]+)", trunk)
    if tag is not None and in_git:
        name, prefix = tag.group(1), tag_prefix(root)
        if prefix and not name.startswith(prefix):
            errors.append("主干 tag {} 应以 {} 开头：同一仓库里的子项目各用各的前缀。".format(name, prefix))
        code, tagged = git(root, "rev-parse", "-q", "--verify", "refs/tags/{}^{{commit}}".format(name))
        if code != 0:
            errors.append("STATE.md 里的主干 tag {} 不存在：先打 tag。".format(name))
        elif match is not None:
            pinned = git(root, "rev-parse", "-q", "--verify", match.group(1) + "^{commit}")[1].strip()
            if pinned and pinned != tagged.strip():
                errors.append("主干 tag {} 指向的提交和 STATE.md 里的 commit {} 不一致。".format(name, match.group(1)))
    if PLACEHOLDER in "\n".join(section(state, "当前唯一问题")):
        warnings.append("STATE.md 没有写当前唯一问题。")
    queued = [line for line in section(state, "排队") if re.match(r"\s*(-|\d+\.)\s+\S", line)]
    if len(queued) > MAX_QUEUE:
        errors.append("STATE.md 排队有 {} 项，上限 {} 项。".format(len(queued), MAX_QUEUE))
    updated = re.search(r"更新[:：]\s*(\d{4}-\d{2}-\d{2})", "\n".join(state))
    if updated is None:
        errors.append("STATE.md 缺少“更新：YYYY-MM-DD”。")
    elif in_git:
        last = git(root, "log", "-1", "--format=%cd", "--date=short", "--", ".")[1].strip()
        if last and updated.group(1) < last:
            warnings.append("STATE.md 更新于 {}，早于本项目最近一次提交 {}：确认它没有过期。".format(updated.group(1), last))

    rows = registry_rows(root)
    experiments = root / "experiments"
    directories = sorted(p for p in experiments.iterdir() if p.is_dir()) if experiments.is_dir() else []
    for directory in directories:
        if not (directory / "CARD.md").is_file():
            errors.append("experiments/{} 没有 CARD.md。".format(directory.name))
        if directory.name not in rows:
            errors.append("experiments/{} 没有登记在 EXPERIMENTS.md。".format(directory.name))
        figures = [p for p in (directory / "figures").glob("*") if p.is_file()] if (directory / "figures").is_dir() else []
        if len(figures) > MAX_FIGURES:
            warnings.append("experiments/{} 有 {} 张图，上限 {} 张。".format(directory.name, len(figures), MAX_FIGURES))
        status = rows.get(directory.name, [""] * 7)[4]
        if status in ("done", "stopped") and not (directory / "result.json").is_file():
            errors.append("experiments/{} 状态是 {}，但没有 result.json。".format(directory.name, status))
    running = [key for key, cells in rows.items() if cells[4] == "running"]
    if len(running) > 1:
        warnings.append("有 {} 个实验同时是 running：{}。同一时间只回答一个问题。".format(len(running), "、".join(running)))

    extra = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in ROOT_DIRS)
    for name in extra:
        errors.append("根目录多出 {}/：代码放进 src/，实验放进 experiments/<id>/，不为一个阶段或实验新建目录。".format(name))
    for relative in files:
        if relative.endswith(".py") and relative.startswith(("src/", "experiments/")) and not relative.endswith("/artifacts.py"):
            if re.search(r"\btorch\.save\(", (root / relative).read_text(encoding="utf-8", errors="replace")):
                warnings.append("{} 直接调用了 torch.save：checkpoint 改用 artifacts.save_checkpoint，缓存改用 artifacts.cache_dir。".format(relative))
    if not getattr(args, "skip_audit", False):
        report = collect(root, duplicates=False)
        removable = sum(size for _, size in report["checkpoints"] + report["misplaced"])
        if removable > AUDIT_WARN_GB * GB:
            warnings.append("runs/ 和 experiments/ 里有 {:.1f}G 冗余产物：运行 tools/project.py audit。".format(removable / GB))
        for run, size, budget in report["over_budget"]:
            warnings.append("{} 用了 {:.1f}G，超出卡片的存储预算 {}G。".format(show(root, run), size / GB, budget))

    for message in errors:
        print("ERROR " + message)
    for message in warnings:
        print("WARN  " + message)
    print("check：{} 个 ERROR，{} 个 WARN。".format(len(errors), len(warnings)))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command")
    commands.required = True
    init = commands.add_parser("init", help="从模板创建项目，只创建缺失的文件")
    init.add_argument("dir")
    init.add_argument("--name", required=True, help="Python 包名，用于 src/<name>/")
    init.add_argument("--title", help="项目显示名，默认同 --name")
    init.add_argument("--date")
    init.set_defaults(run=cmd_init)
    new = commands.add_parser("new", help="新建实验卡片并登记")
    new.add_argument("slug")
    new.add_argument("--type", required=True, choices=TYPES)
    new.add_argument("--date")
    new.add_argument("--root", default=".")
    new.set_defaults(run=cmd_new)
    check = commands.add_parser("check", help="检查项目是否符合模板约定")
    check.add_argument("--root", default=".")
    check.add_argument("--skip-audit", action="store_true", help="不统计 runs/ 的冗余产物（runs/ 很大时用）")
    check.set_defaults(run=cmd_check)
    audit = commands.add_parser("audit", help="只读：找出冗余 checkpoint、放错位置的缓存、超预算和重复文件，生成删除清单")
    audit.add_argument("--exp", help="只看这一个实验")
    audit.add_argument("--runs", action="append", default=[], help="另一个 runs 根目录，可重复")
    audit.add_argument("--root", default=".")
    audit.set_defaults(run=cmd_audit)
    prune = commands.add_parser("prune", help="删除清单里的路径；不加 --yes 只预览")
    prune.add_argument("list")
    prune.add_argument("--yes", action="store_true", help="用户已确认，执行删除")
    prune.add_argument("--runs", action="append", default=[], help="允许删除的另一个 runs 根目录")
    prune.add_argument("--root", default=".")
    prune.set_defaults(run=cmd_prune)
    args = parser.parse_args()
    sys.exit(args.run(args) or 0)


if __name__ == "__main__":
    main()
