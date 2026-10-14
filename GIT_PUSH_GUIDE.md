# 推送到 GitHub 操作说明

本地仓库已就绪，**仅剩远程推送这一步需要你提供信息**。

## 当前状态

| 项目 | 状态 |
|---|---|
| 本地仓库 | 已初始化（分支 `main`） |
| 首次提交 | 已完成 `c4ccfe0`，共 23 个文件 |
| 远程 `origin` | **未配置** |
| 推送 | **未执行** |

## 你需要提供的两项信息

1. **仓库地址**：形如 `https://github.com/用户名/仓库名.git`
   （若仓库尚未创建，先在 GitHub 上新建一个空仓库，**不要**勾选 README/.gitignore/LICENSE）
2. **Personal Access Token**：用于非交互式推送
   - 生成路径：`GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token`
   - 勾选权限：`repo`（私有仓库必需）
   - 有效期建议 90 天，过期需重新生成

## 拿到后的执行命令

```bash
cd C:/Users/jsjjlt/WorkBuddy/2026-10-14-16-55-02

# 1. 添加远程地址
git remote add origin <你的仓库地址>

# 2. 推送并建立上游追踪
git push -u origin main
```

推送时若提示输入凭据：

- Username：你的 GitHub 用户名
- Password：**粘贴 Token**，不是登录密码

> 认证失败会缓存错误凭据，需先清除：
> `git config --global credential.helper store` 配置错误时执行
> `cmdkey /delete:git:https://github.com`

## 或者交给 WorkBuddy 代办

直接把仓库地址和 Token 发给我，我执行以下命令完成推送并校验：

```bash
git remote add origin <url>
git push -u origin main
python .workbuddy/skills/git-sync-required/scripts/sync.py --check
```

> 说明：Token 属于敏感凭据。若不希望在对话中传递，可自行在本地执行上述命令。`.env`、`*.pem`、`token.txt` 等文件已由 `.gitignore` 拦截，不会误入库。

## 后续强制同步

项目已启用 `git-sync-required` skill，今后每次修改代码或配置后必须提交并推送。

日常一键同步：

```bash
python .workbuddy/skills/git-sync-required/scripts/sync.py "feat: 本次改动说明"
```

该脚本会自动完成暂存、提交、推送，并校验本地与 `origin/main` 是否一致。

单独检查工作区是否干净：

```bash
python .workbuddy/skills/git-sync-required/scripts/sync.py --check
```

## 仓库内容说明

**已纳入版本控制（23 个文件）**

- 13 个研究脚本：`fetch_data.py`、`build_features_v2.py`、`train_model.py`、`transformer_model.py`、`v2_experiment.py`、`strict_eval.py`、`significance_test.py`、`attribution.py`、`analyze_factors.py`、`diagnose_alpha.py` 等
- 2 个预注册文件：`PREREGISTRATION.json`、`PREREGISTRATION_V2.json`
- 5 个实验结果摘要：`output/` 下的 `strict_results.json`、`significance_test.json`、`analysis_summary.json`、`attribution.json`、`factor_correlations.csv`
- 3 个数据元信息：`data/meta.json`、`data/meta_v2.json`、`data/nfp_months.json`
- 2 个规范文件：`.gitignore` 与项目级 skill

**已排除（可复现，不占仓库体积）**

- `*.npy`、`*.pt` 等模型与预测向量产物
- `data/gold_features*.csv`、`gold_raw.csv`（约 12MB）
  → 复现命令：依次运行 `python fetch_data.py`、`python build_features_v2.py`
- `.workbuddy/memory/` 等本地运行数据
