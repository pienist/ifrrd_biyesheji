class ConvNeXtConfig:
    """ConvNeXt 模型配置类"""
    
    # 模型预设配置
    CONFIGS = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768],
            'params': 29,  # 百万级
            'flops': 4.5   # 十亿级
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'dims': [96, 192, 384, 768],
            'params': 50,
            'flops': 8.7
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'dims': [128, 256, 512, 1024],
            'params': 89,
            'flops': 15.4
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'dims': [192, 384, 768, 1536],
            'params': 306,
            'flops': 52.8
        }
    }
    
    @staticmethod
    def get_config(model_type: str):
        """获取指定模型的配置"""
        if model_type not in ConvNeXtConfig.CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        return ConvNeXtConfig.CONFIGS[model_type]


def initialize_model(model_type: str = 'small', pretrained: bool = False):
    """
    初始化 ConvNeXt 模型
    
    参数：
        model_type: 模型类型 ('tiny', 'small', 'base', 'large')
        pretrained: 是否加载 ImageNet 预训练权重
        
    返回：
        model: 初始化后的模型对象
    """
    if model_type == 'tiny':
        model = convnext_tiny()
    elif model_type == 'small':
        model = convnext_small()
    elif model_type == 'base':
        model = convnext_base()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 如果需要加载预训练权重
    if pretrained:
        print(f"加载 {model_type} 模型的预训练权重...")
        # 这里应该从官方源下载预训练权重
        # 实现细节依赖于具体的权重存储位置
        pass
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数总数和可训练参数数
    
    参数：
        model: 神经网络模型
        
    返回：
        (total_params, trainable_params): 总参数和可训练参数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def calculate_flops(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """
    计算模型的浮点运算数 (FLOPs)
    
    参数：
        model: 神经网络模型
        input_size: 输入张量大小 (C, H, W)
        
    返回：
        flops: 浮点运算数（以 GMac 为单位）
    """
    from functools import reduce
    from operator import mul
    
    def conv_flops(m, input, output):
        """计算卷积层的 FLOPs"""
        batch_size = input[0].size(0)
        output_height, output_width = output[0].size(2), output[0].size(3)
        kernel_dims = list(m.kernel_size)
        in_channels = m.in_channels
        out_channels = m.out_channels
        groups = m.groups
        
        filters_per_channel = out_channels // groups
        conv_per_position_flops = reduce(mul, kernel_dims) * (in_channels // groups)
        active_elements_count = batch_size * output_height * output_width
        overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
        
        bias_flops = 0
        if m.bias is not None:
            bias_flops = out_channels * active_elements_count
        
        return overall_conv_flops + bias_flops
    
    def linear_flops(m, input, output):
        """计算线性层的 FLOPs"""
        batch_size = input[0].size(0)
        return batch_size * input[0].size(1) * output[0].size(1)
    
    model_flops = 0
    
    def add_hooks(m):
        """添加钩子函数"""
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(lambda m, inp, out: conv_flops(m, inp, out))
        if isinstance(m, nn.Linear):
            m.register_forward_hook(lambda m, inp, out: linear_flops(m, inp, out))
    
    model.apply(add_hooks)
    
    # 进行一次前向传播
    try:
        dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
        _ = model(dummy_input)
    except Exception as e:
        print(f"计算 FLOPs 时出错: {e}")
        return 0
    
    return model_flops


def print_model_summary(model: nn.Module, model_type: str = 'small'):
    """
    打印模型摘要信息
    
    参数：
        model: 神经网络模型
        model_type: 模型类型
    """
    total_params, trainable_params = count_parameters(model)
    config = ConvNeXtConfig.get_config(model_type)
    
    print("=" * 70)
    print(f"ConvNeXt-{model_type.upper()} 模型摘要")
    print("=" * 70)
    print(f"总参数数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"预设参数数: {config['params']}M")
    print(f"预设 FLOPs: {config['flops']}B")
    print("=" * 70)


# ============================================================================
# 代码解析
# ============================================================================
"""
工具函数的核心作用：

1. count_parameters():
   - 统计模型参数总数
   - 区分可训练和冻结参数
   - 用于模型复杂度分析
   
   实现原理：
   model.parameters() 返回所有可学习参数
   p.numel() 返回参数元素总数
   requires_grad 标记参数是否可训练

2. calculate_flops():
   - 通过 forward hooks 计算浮点运算数
   - Conv2d: kernel_size × in_channels × H × W × out_channels
   - Linear: batch_size × in_features × out_features
   
   注意：
   - 仅为估算值，实际会因硬件而异
   - 不包括激活函数的计算
   - 不包括归一化层的计算

3. print_model_summary():
   - 对模型信息进行格式化输出
   - 便于快速了解模型规模
   - 与官方预设配置对比
"""
