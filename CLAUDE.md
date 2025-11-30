# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PISA 行為分析後端專案 - 分析 SPSS 格式的 PISA 學生行為資料，轉換為 Parquet 格式並使用 PySpark 進行統計分析。

## Development Environment

### Package Management

使用 **uv** 作為 Python 套件管理工具（非 pip/poetry）：

```bash
uv sync --all-groups              # 安裝所有依賴（含開發依賴）
uv run python -m behavior_analysis  # 執行主程式
```

### Docker Environment

開發環境運行在 Docker 容器中：

```bash
docker-compose up -d --build      # 啟動 JupyterLab 環境
docker-compose logs jupyter       # 查看登入連結
```

使用 VSCode Remote Container 連接到 `bdm_student_behavior_backend_jupyter_1` 容器進行開發。**如果你在容器中，請確保使用 `conda activate python310` 後再來執行命令。**

## Code Quality & Pre-commit

### 必須遵守的規範

本專案使用嚴格的程式碼品質檢查，**所有變更都必須通過**：

```bash
# 安裝 pre-commit hooks（初次設定）
uv run pre-commit install

# 執行所有檢查
uv run pre-commit run --all-files
```

### 關鍵規則

- 每次完成修改前必須執行 pre-commit 檢查
- agent 位於容器中時使用 `conda activate python310` 啟動環境，容器環境沒有安裝 pre-commit，請提示使用者在本地機器執行pre-commit。
- agent 在本地機器時，直接使用 `uv run pre-commit run --all-files` 執行檢查，使用 `docker exec -it bdm_student_behavior_backend-jupyter-1 bash` 進入容器執行程式。

## Architecture

### 三層架構

```
behavior_analysis/
├── __main__.py          # 應用程式入口點（workflow orchestration）
├── config.py            # 集中式配置（DataConfig, SparkConfig, ConversionConfig）
├── data/                # 資料層
│   ├── converter.py     # SPSS → Parquet 轉換器（支援大檔案批次處理）
│   ├── spark_manager.py # Spark Session 單例管理器（context manager）
│   └── validator.py     # 資料驗證工具
├── analysis/            # 分析層
│   └── basic_stats.py   # 統計分析函數（使用 PySpark）
└── utils/               # 工具層
    ├── logger.py        # 集中式日誌管理
    └── file_utils.py    # 檔案操作工具
```

### 主要工作流程（**main**.py）

1. **初始化** → 載入配置、設定日誌
2. **轉換階段** → SPSS 轉 Parquet（使用 `SPSSToParquetConverter`）
3. **Spark 分析** → 載入 Parquet、執行統計分析
4. **結果輸出** → 格式化統計報告

### 重要設計模式

#### Singleton Pattern

`SparkSessionManager` 使用單例模式確保只有一個 Spark session：

```python
with SparkSessionManager(config.spark) as spark:
    # Spark operations
    # Session automatically cleaned up on exit
```

#### 批次處理策略

`SPSSToParquetConverter` 根據檔案大小自動選擇策略：

- 小檔案（< 500MB）：一次性載入記憶體
- 大檔案（≥ 500MB）：批次處理（預設 100,000 rows/batch）

#### 轉換狀態追蹤

使用隱藏檔案追蹤轉換狀態（`.{filename}.parquet.converted`）：

- 包含時間戳、檔案大小、修改時間等 metadata
- 自動檢測 SPSS 檔案更新並重新轉換

### Configuration System

使用 dataclass 實現階層式配置：

```python
config = load_config()
config.data.DATA_DIR         # 資料目錄
config.spark.DRIVER_MEMORY   # Spark 記憶體配置
config.conversion.CHUNK_SIZE # 批次大小
```

環境變數覆蓋（可選）：

- `BA_DATA_DIR` → `config.data.DATA_DIR`
- `BA_SPARK_DRIVER_MEMORY` → `config.spark.DRIVER_MEMORY`

### Spark 最佳化配置

針對本地環境（19GB RAM）優化：

- Driver Memory: 6GB
- Executor Memory: 8GB
- Cores: 4（避免過度平行化）
- Shuffle Partitions: 20（降低 overhead）
- Adaptive Execution: Enabled

## Running the Application

### 基本執行

```bash
uv run python -m behavior_analysis
```

### 資料集配置

預設處理三個 SPSS 檔案（在 `config.py` 的 `DataConfig.SPSS_FILES`）：

- `student`: CY08MSP_STU_QQQ.SAV
- `teacher`: CY08MSP_TCH_QQQ.SAV
- `school`: CY08MSP_SCH_QQQ.SAV

## Testing & CI/CD

### GitHub Actions

PR 時自動執行（`.github/workflows/ci.yml`）：

- Ruff linting & formatting
- Mypy type checking
- Pre-commit hooks
- 支援 Python 3.10, 3.11, 3.12

### 本地測試流程

```bash
uv run pre-commit run --all-files  # 執行所有品質檢查
```

## Data Flow

```
SPSS Files (.SAV)
    ↓ [SPSSToParquetConverter]
Parquet Files (.parquet)
    ↓ [SparkSessionManager + spark.read.parquet()]
Spark DataFrame
    ↓ [analysis/basic_stats.py]
Statistical Results
    ↓ [print_statistics_report()]
Formatted Output（console/file）
```

## Performance Considerations

- **記憶體管理**：大檔案使用批次處理避免 OOM
- **磁碟空間**：轉換前檢查可用空間（預留 20% buffer）
- **Parquet 壓縮**：預設使用 snappy（平衡速度與壓縮率）
- **Spark 優化**：已針對本地執行調整，避免修改 `SparkConfig` 預設值

## File Naming Conventions

- Python 模組：snake_case
- Class：PascalCase
- 函數/變數：snake_case
- 常數：UPPER_SNAKE_CASE（在 dataclass 中）
- Private：前綴 `_`（如 `_initialized`, `_session`）
