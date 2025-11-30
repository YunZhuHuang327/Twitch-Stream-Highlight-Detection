# Highlight Detection 训练脚本

# 设置环境变量
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

# 激活 conda 环境
conda activate chapter-llama

# 运行训练
python train.py `
    experiment=highlight `
    data=highlight `
    model=llama3.2_1B_highlight `
    logger.project=highlight_detection `
    data.train_subset=train `
    data.val_subset=val

Write-Host "`n训练完成！" -ForegroundColor Green
