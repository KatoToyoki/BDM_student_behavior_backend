# BDM_student_behavior_backend

## Setup Instructions

執行 `docker-compose up -d --build` 啟動服務，服務啟動後，使用 `docker-compose logs jupyter` 查看 JupyterLab logs，找到登入連結。

### vscode remote container

為方便開發，也可以使用 vscode remote container 功能，點擊畫面左下角的綠色圖示，選擇 「attach to running container」，選擇 `bdm_student_behavior_backend_jupyter_1` 即可進入容器開發環境。

## Development Setup

### 安裝開發依賴

本專案使用 [uv](https://github.com/astral-sh/uv) 作為 Python 套件管理工具。

```bash
# 安裝所有依賴（包含開發依賴）
uv sync --all-groups
```

### Pre-commit Hooks

本專案使用 pre-commit 確保程式碼品質，在每次 commit 前自動執行 linting 和 type checking。

```bash
# 安裝 pre-commit hooks
uv run pre-commit install

# 手動執行所有 hooks
uv run pre-commit run --all-files
```

Pre-commit 會自動執行以下檢查:

- **Ruff**: Python linter 和 code formatter
- **Mypy**: 靜態型別檢查
- 其他基本檢查（trailing whitespace、file endings 等）

### Code Quality Tools

#### Ruff

Ruff 是一個快速的 Python linter 和 formatter。

```bash
# 執行 linting
uv run ruff check .

# 自動修復問題
uv run ruff check --fix .

# 執行 code formatting
uv run ruff format .
```

#### Mypy

Mypy 用於靜態型別檢查。

```bash
# 執行型別檢查
uv run mypy behavior_analysis
```

### CI/CD

本專案使用 GitHub Actions 進行持續整合，在每次 PR 時會自動執行:

- Ruff linting
- Ruff formatting check
- Mypy type checking
- Pre-commit hooks

請確保本地通過所有檢查後再提交 PR。
