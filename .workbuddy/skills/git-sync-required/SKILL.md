---
name: git-sync-required
description: 本项目（量化因子研究 / NFP 金价预测实验）实行强制版本同步纪律。任何对项目文件的创建、修改、删除，在任务收尾时必须执行 git add + commit + push，否则视为任务未完成。此 skill 应在本项目内每次编写或改动代码、配置、预注册文件、文档后触发，强制完成版本控制闭环。
agent_created: true
---

# 项目强制 Git 同步规范

## 目的

本项目的研究结论具有时效性，代码与预注册文件（`PREREGISTRATION*.json`）共同构成研究可信度证据链。任何未被提交的改动都可能造成证据链断裂或实验结果无法复现，因此本项目采用**"改动必须入库"**的硬性纪律。

## 适用场景

在本仓库内执行以下任一操作后，必须触发本 skill：

- 创建、修改、删除任何 `.py` 脚本、`.json` 配置、`PREREGISTRATION*.json`、`.md` 文档
- 修改 `data/` 或 `output/` 下的小体积产物（`.json`、`.csv` 摘要类）
- 执行实验、重跑分析、生成新的统计结果
- 用户提示"提交"、"同步"、"上传"、"保存版本"、"commit"、"push"

## 核心规则（不可协商）

1. **不允许留下未提交的改动。** 任务结束时 `git status` 必须显示 working tree clean。
2. **不允许只 commit 不 push。** 提交必须推送到 `origin`（`main` 分支），除非远程尚未配置。
3. **不允许 `--force` 强推 main 分支。** 历史改写仅在用户明确要求时执行。
4. **不允许提交密钥与凭据。** `.env`、`*.pem`、`token.txt` 等已在 `.gitignore` 中拦截，不得绕过。
5. **不允许提交大体量中间文件。** `*.npy`、`*.pt`、`data/*.csv` 特征矩阵由脚本可复现，保持忽略。

## 标准执行流程

在仓库根目录（`C:/Users/jsjjlt/WorkBuddy/2026-10-14-16-55-02`）按顺序执行：

```bash
# 1. 先看清将要提交什么，绝不使用 git add .
git status --short
git diff --stat

# 2. 精确暂存本任务涉及的改动
git add <具体文件路径...>

# 3. 再次确认暂存内容
git status --short

# 4. 提交（message 需说明本次改动做了什么，而非泛泛的 "update"）
git commit -m "简明描述本次改动"

# 5. 推送到远程 main
git push origin main
```

或使用本 skill 提供的自动化脚本完成全流程：

```bash
python .workbuddy/skills/git-sync-required/scripts/sync.py "本次改动的说明"
```

脚本会自动暂存全部跟踪文件、提交、推送，并输出最终校验状态。

## Commit Message 规范

采用祈使句、说明实际变更，不写空话。推荐格式：

```
<类型>: <简明描述>

类型取值：
  feat     新增功能或脚本
  fix      修复缺陷
  refactor 重构，不改变行为
  data     更新数据或实验产物
  docs     文档与规范
  config   配置与预注册文件
```

示例：

- `feat: 新增 v2 特征构建流程与 transformer 模型`
- `fix: 修正严格评估中的未来收益对齐偏移`
- `config: 更新 V2 预注册方案的窗口参数`
- `data: 记录显著性检验与因子相关性结果`

## 完成判据

同步完成需同时满足：

- [ ] `git status --short` 输出为空
- [ ] `git log --oneline -1` 显示本次提交
- [ ] `git push` 返回成功，或已明确告知用户远程尚未配置、需补充地址与凭据

三项中任何一项不满足，都必须在最终回复中显式说明，不得静默跳过。

## 边界情况处理

**远程未配置时**：完成本地 `git add` 与 `git commit`，在回复中明确提示用户提供 GitHub 仓库地址与 Personal Access Token，切勿声称已上传。

**推送被拒绝（非 fast-forward）**：先执行 `git fetch origin` 与 `git status`，若为分叉，汇报冲突情况并询问用户选择 rebase 还是 merge，不得擅自决定。

**改动只涉及被忽略的文件**：无需提交，但需在回复中说明哪些文件因体积或性质未纳入版本控制，以及如何复现。

**回滚需求**：使用 `git revert` 生成新提交，不使用 `git reset --hard` 破坏历史。

## 验证自检

任务收尾前执行：

```bash
git status --short          # 应为空
git log --oneline -3        # 确认提交记录
git log origin/main -1 --oneline  # 远程已有记录
```

若本地 HEAD 与 `origin/main` 不一致，说明推送未成功，必须重新推送或告知用户。

## 资源

- `scripts/sync.py` —— 一键完成暂存、提交、推送与状态校验的自动化脚本
