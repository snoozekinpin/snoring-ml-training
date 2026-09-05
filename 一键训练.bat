@echo off
chcp 65001 >nul
echo ============================================================
echo   酣眠 SnoozMate - 鼾声检测模型训练
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查是否需要安装依赖
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
)

echo [2/4] 安装依赖...
call .venv\Scripts\pip install -r requirements.txt -q
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，尝试继续...
)

REM 检查数据集
if not exist "dataset\snore\*.wav" (
    echo.
    echo [3/4] 数据集目录为空，生成合成测试数据...
    call .venv\Scripts\python download_datasets.py --synth --snore_count 50 --noise_count 50
) else (
    echo [3/4] 数据集已存在
)

echo.
echo [4/4] 开始训练...
echo.

set EPOCHS=30
if "%1"=="" goto default_epochs
set EPOCHS=%1
:default_epochs

call .venv\Scripts\python train.py --epochs %EPOCHS% --batch_size 32

echo.
echo ============================================================
echo   训练完成！输出在 output\model\ 目录
echo   ESP32 部署文件在 output\esp32\snore_model.h
echo ============================================================
echo.
pause
