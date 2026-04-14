#!/usr/bin/env python3
"""
YOLOv11-ConvNeXt 物体检测完整示例

该文件展示了如何使用 YOLOv11-ConvNeXt 模型进行：
- 单张图像检测
- 检测结果可视化
- 视频处理
- 实时网络摄像头检测
"""

import time
from tqdm import tqdm


class ObjectDetectionDemo:
    """
    使用 YOLOv11-ConvNeXt 进行物体检测的完整示例
    """
    
    def __init__(self, model_path: str):
        """初始化检测器"""
        self.detector = YOLOv11ConvNeXtInference(
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half_precision=torch.cuda.is_available()
        )
        
        # COCO 类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def visualize_detections(
        self,
        image_path: str,
        detections,
        output_path: str = None
    ):
        """
        可视化检测结果
        
        参数：
            image_path: 图像路径
            detections: 检测结果
            output_path: 输出路径
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # 颜色配置
        colors = {
            'person': '#FF6B6B',
            'car': '#4ECDC4',
            'dog': '#45B7D1',
            'cat': '#96CEB4',
            'default': '#FFEAA7'
        }
        
        # 绘制检测框
        for detection in detections:
            # 提取信息
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # 选择颜色
            color = colors.get(class_name, colors['default'])
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1-10), label, fill=color)
        
        # 保存或显示
        if output_path:
            image.save(output_path)
            print(f"可视化结果已保存到 {output_path}")
        
        return image
    
    def detect_video(self, video_path: str, output_path: str = None):
        """
        视频检测
        
        参数：
            video_path: 视频文件路径
            output_path: 输出视频路径
        """
        import cv2
        
        print(f"\n处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"分辨率: {frame_width}×{frame_height}")
        print(f"帧率: {fps} FPS")
        print(f"总帧数: {total_frames}\n")
        
        # 视频写入器
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width, frame_height))
        
        frame_count = 0
        total_time = 0.0
        
        with tqdm(total=total_frames, desc="视频处理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存临时图像
                temp_image = '/tmp/temp_frame.jpg'
                cv2.imwrite(temp_image, frame)
                
                # 运行检测
                detections, inference_time = self.detector.inference(temp_image)
                total_time += inference_time
                
                # 绘制结果
                # (这里需要实现具体的绘制逻辑)
                
                # 写入输出视频
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"\n视频处理完成")
        print(f"处理帧数: {frame_count}")
        print(f"平均推理时间: {total_time/frame_count*1000:.2f}ms")
        print(f"实时处理性能: {frame_count/total_time:.1f} FPS")
    
    def detect_webcam(self, duration: int = 30):
        """
        实时网络摄像头检测
        
        参数：
            duration: 运行时长(秒)
        """
        import cv2
        
        print(f"\n启动网络摄像头检测({duration}秒)")
        
        cap = cv2.VideoCapture(0)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"分辨率: {frame_width}×{frame_height}\n")
        
        start_time = time.time()
        frame_count = 0
        total_inference_time = 0.0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存临时图像
            temp_image = '/tmp/temp_webcam.jpg'
            cv2.imwrite(temp_image, frame)
            
            # 运行检测
            detections, inference_time = self.detector.inference(temp_image)
            total_inference_time += inference_time
            
            # 计算 FPS
            fps = frame_count / (time.time() - start_time)
            
            # 绘制 FPS
            # (这里需要实现具体的绘制逻辑)
            
            frame_count += 1
            
            # 显示帧
            # cv2.imshow('Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n网络摄像头检测完成")
        print(f"处理帧数: {frame_count}")
        print(f"平均推理时间: {total_inference_time/frame_count*1000:.2f}ms")
        print(f"实时处理性能: {frame_count/total_inference_time:.1f} FPS")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    # 初始化演示对象
    demo = ObjectDetectionDemo(model_path='results/models/best.pt')
    
    # 示例1: 单张图像检测
    # detections = demo.detector.inference('path/to/image.jpg')
    # demo.visualize_detections('path/to/image.jpg', detections, 'output.jpg')
    
    # 示例2: 视频检测
    # demo.detect_video('path/to/video.mp4', 'output_video.mp4')
    
    # 示例3: 网络摄像头实时检测
    # demo.detect_webcam(duration=60)
    
    print("请取消注释上面的示例代码来使用")


# ============================================================================
# 代码解析
# ============================================================================
"""
实际应用示例的核心设计：

1. 可视化检测结果：
   
   要素：
   ① 边界框：绘制矩形框
   ② 类别标签：显示物体类别
   ③ 置信度：显示检测置信度
   ④ 颜色编码：不同类别用不同颜色
   
   实现技巧：
   - 使用 PIL 进行图像绘制
   - 颜色使用 16 进制码
   - 字体大小根据图像尺寸调整
   - 标签位置避免重叠

2. 视频处理流程：
   
   步骤：
   ① 打开视频文件获取参数
   ② 按帧读取并处理
   ③ 每帧运行目标检测
   ④ 绘制检测结果
   ⑤ 写入输出视频
   
   性能考虑：
   - 帧间隔调整处理速度
   - 关键帧检测优化
   - 视频缓冲管理

3. 实时网络摄像头检测：
   
   关键指标：
   ① 实时 FPS：显示每秒处理帧数
   ② 延迟：检测耗时
   ③ GPU 利用率：推理效率
   
   优化策略：
   - 批处理多帧
   - 自适应分辨率
   - 异步处理队列
   - 结果缓存机制

4. 性能监控：
   
   要监控的指标：
   ① 推理时间分布
   ② CPU/GPU 使用率
   ③ 内存占用
   ④ 热度和功耗
"""
