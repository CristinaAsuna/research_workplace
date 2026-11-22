# research_workplace：总仓库规则

本仓库装多个研究子项目，每个子项目一个目录。本文件只管子项目之间的关系；子项目自己的 AGENTS.md 管其余一切，并且优先于本文件。

## 结构

| 目录 | 放什么 |
| --- | --- |
| `paper_repro/<名称>/` | 复现一篇论文 |
| `idea_verify/<名称>/` | 验证自己的想法 |
| `README.md` | 子项目索引，每个子项目一行 |
| `.lab/runs/` | 所有子项目的运行记录，由 `lab` 命令维护，不手改 |

根目录只有上面这些和 AGENTS.md、CLAUDE.md、.gitignore。不在根目录放代码、数据或文档。

## 规则

1. **一次只做一个子项目。** 在子项目目录里启动会话。在根目录启动时，先问用户做哪个子项目，不通读其他子项目。
2. **skill 按路径读。** 子项目按 skill `research-project` 和 `lab-machines` 管理。它们不会自动加载：需要时直接读 `~/.agents/skills/<名称>/SKILL.md`。
3. **先查已有的数据和权重。** 写 DATA.md、计划下载或预处理之前，先读总台的 MACHINES.md 资产表（总台路径见 `~/.config/lab/hub`），再到目标机器上 `ls` 它的数据目录（集群是 `/public_bme/home/maxquan/Datasets/`）。两处都没有，才计划下载；找到了没登记的，先补登记。新数据集放 `$LAB_DATA_DIR/<名称>/`，权重放 `$LAB_MODELS_DIR`，都不放进子项目。
4. **子项目互不依赖。** 不 import 其他子项目的代码，不读它们的 runs/。想共用代码时先问用户。
5. **新建子项目**：在仓库根目录运行
   `python ~/.agents/skills/research-project/scripts/project.py init <paper_repro|idea_verify>/<名称> --name <包名>`，
   然后和用户一起填它的 docs/ 和 STATE.md，并在 README.md 的索引里加一行。主干 tag 自动带子项目路径前缀。
6. **运行和留痕**：在子项目目录里用 `lab run start/end`，代码快照和产物落在该子项目的 `runs/`。运行、下载、安装、同步、删除之前，先告诉用户"机器 / 目录 / 命令 / 预计耗时与显存 / 读写哪些数据"，等确认后再执行。
7. **结束**：提交，再 `lab sync`。
