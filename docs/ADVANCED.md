# 训练模型
python train.py --model convnext-small --epochs 100 --batch-size 32

# 推理单图像
python inference.py --image test.jpg --model best_model.pth

# 性能基准
python benchmark.py --model convnext-small --num-iterations 100
