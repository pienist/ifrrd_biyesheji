class ConvNeXtTrainer:
    """
    ConvNeXt 模型训练器
    
    功能：
    1. 模型初始化和参数冻结
    2. 优化器和学习率调度
    3. 训练循环和验证
    4. 检查点保存和加载
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 20
    ):
        """
        初始化训练器
        
        参数：
            model: 要训练的模型
            device: 设备类型 ('cuda' 或 'cpu')
            lr: 初始学习率
            weight_decay: 权重衰减系数
            warmup_epochs: 预热周期数
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
        # 优化器：使用 AdamW 比标准 Adam 更稳定
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 损失函数记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def freeze_backbone(self, freeze: bool = True):
        """
        冻结或解冻骨干网络参数
        
        参数：
            freeze: True 为冻结，False 为解冻
            
        用途：
        1. 迁移学习中冻结预训练权重
        2. 只微调特定层
        3. 降低计算成本
        """
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'stem' in name:
                param.requires_grad = not freeze
                
        # 重新创建优化器，只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        status = "冻结" if freeze else "解冻"
        print(f"✓ 骨干网络已{status}")

    def get_lr_scheduler(self, total_epochs: int):
        """
        获取学习率调度器
        
        参数：
            total_epochs: 总训练轮数
            
        返回：
            scheduler: 学习率调度器对象
            
        策略：
        1. 预热阶段（线性增长）
        2. 衰减阶段（余弦衰减）
        """
        def lr_lambda(current_epoch):
            # 预热阶段
            if current_epoch < self.warmup_epochs:
                return current_epoch / self.warmup_epochs
            
            # 衰减阶段：使用余弦衰减
            progress = (current_epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def train_epoch(self, train_loader, criterion):
        """
        训练一个完整的 epoch
        
        参数：
            train_loader: 训练数据加载器
            criterion: 损失函数
            
        返回：
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="训练中", leave=False)
        
        for images, targets in progress_bar:
            # 移动数据到设备
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self, val_loader, criterion):
        """
        验证模型性能
        
        参数：
            val_loader: 验证数据加载器
            criterion: 损失函数
            
        返回：
            avg_loss: 平均验证损失
            accuracy: 准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="验证中", leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                # 记录损失
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples
        
        self.val_losses.append(avg_loss)
        
        # 如果是最佳模型，保存
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best_model.pth')
        
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs: int, criterion):
        """
        完整的训练流程
        
        参数：
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            criterion: 损失函数
        """
        scheduler = self.get_lr_scheduler(epochs)
        
        print(f"\n{'='*70}")
        print(f"{'开始训练 ConvNeXt 模型':^70}")
        print(f"{'='*70}")
        print(f"{'训练轮数':20} {epochs}")
        print(f"{'初始学习率':20} {self.lr}")
        print(f"{'权重衰减':20} {self.weight_decay}")
        print(f"{'预热周期':20} {self.warmup_epochs}")
        print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, criterion)
            
            # 验证
            val_loss, accuracy = self.validate(val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印进度
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Accuracy:   {accuracy:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}\n")

    def save_checkpoint(self, checkpoint_path: str):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, checkpoint_path)
        print(f"✓ 检查点已保存到 {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"✓ 检查点已加载自 {checkpoint_path}")


# ============================================================================
# 代码解析
# ============================================================================
"""
ConvNeXtTrainer 的设计要点：

1. 冻结骨干网络 (freeze_backbone):
   - 迁移学习中的标准操作
   - 保留预训练权重的知识
   - 只优化检测头等任务特定层
   
   何时使用：
   ① 数据较少时：冻结 Backbone，微调检测头
   ② 数据充足时：全量微调所有参数
   ③ 计算资源有限：冻结 Backbone 降低内存使用

2. 学习率调度策略：
   
   预热阶段 (Warmup):
   - 从 0 逐步增加到设定的学习率
   - 避免开始时梯度过大导致训练不稳定
   - 通常持续 20-50 个 epoch
   
   衰减阶段 (Cosine Annealing):
   - 使用余弦函数逐步降低学习率
   - 在训练后期稳定性更好
   - 比线性衰减效果更优
   
   公式：
   Warmup: lr = initial_lr * (epoch / warmup_epochs)
   Cosine: lr = 0.5 * initial_lr * (1 + cos(π * progress))

3. 梯度裁剪 (Gradient Clipping):
   - 限制梯度范数不超过 1.0
   - 防止梯度爆炸
   - 特别在 Transformer 类模型中重要

4. 检查点管理：
   - 保存最佳验证损失的模型
   - 便于中断后恢复训练
   - 包含完整的训练状态
"""
