通用穿越火线cf数据集1.5万张，有labels，yolo格式标注信息，分敌我
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9cb15958a2bc49e7af5ee0af0cfe59ac.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8aa83c81c6684f20a8b2f1586a2f3c02.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/18db2da0efb04b569e01d87df41893d2.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/277f6be80d5c4a2fbff3543fee1d6437.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bfe5426efd87411ab3507edb0622b48e.png)
 **通用穿越火线（CrossFire, CF）游戏场景目标检测数据集** 的完整说明与训练方案，包含：

- ✅ 数据集结构说明  
- ✅ 类别定义（分敌我）  
- ✅ YOLO 格式标注规范  
- ✅ 完整 YOLOv8 训练代码（含 `dataset.yaml`）  
- ✅ 推理与部署示例  

---

## ✅ 一、数据集概览

| 项目 | 内容 |
|------|------|
| **数据集名称** | CrossFire Player Detection Dataset (CF-PD v1.0) |
| **图像总数** | 15,000 张（来自游戏录屏/截图） |
| **标注格式** | YOLO `.txt`（每图对应一个标签文件） |
| **分辨率** | 多为 1920×1080 或 1280×720（训练时统一缩放） |
| **场景类型** | 团队竞技、爆破模式、生化模式等主流地图 |
| **标注工具** | LabelImg / 自研游戏标注工具 |
| **关键特性** | **区分敌我阵营**（基于游戏 HUD 颜色或模型特征） |

---

## ✅ 二、类别定义（共 2 类）

> 🎯 所有目标均为 **玩家角色（Player）**，仅按阵营区分：

| class_id | 类别名称（标签名） | 说明 |
|----------|------------------|------|
| 0 | `enemy` | 敌方玩家（通常头顶红名 / 模型偏暗） |
| 1 | `ally`  | 己方玩家（通常头顶绿名 / 蓝名 / 模型亮色） |

> ⚠️ 注意：
> - 不包含武器、箱子、NPC 等非玩家目标
> - 若需扩展（如“生化幽灵”），可后续增加类别
> - 标注框覆盖 **全身或上半身**（确保头部可见以判别阵营）

---

## ✅ 三、数据集目录结构（YOLO 标准格式）

```bash
cf_dataset/
├── images/
│   ├── train/        # ~12,000 张
│   ├── val/          # ~1,500 张
│   └── test/         # ~1,500 张
└── labels/
    ├── train/
    ├── val/
    └── test/
```

> ✅ 图像格式：`.jpg`  
> ✅ 标签格式示例（`labels/train/0001.txt`）：
```
0 0.45 0.62 0.12 0.35   # enemy
1 0.78 0.55 0.10 0.30   # ally
```

---

## ✅ 四、`dataset.yaml` 配置文件

在项目根目录创建 `cf_dataset/dataset.yaml`：

```yaml
# cf_dataset/dataset.yaml
path: ./cf_dataset
train: images/train
val: images/val
test: images/test

nc: 2
names: ['enemy', 'ally']
```

---

## ✅ 五、YOLOv8 训练代码（`train_cf.py`）

```python
# train_cf.py
from ultralytics import YOLO

def main():
    # 加载预训练模型（YOLOv8n 更适合游戏小目标）
    model = YOLO('yolov8n.pt')  # 可选 yolov8s/m/l

    # 开始训练
    results = model.train(
        data='cf_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,              # 游戏画面细节丰富，建议 ≥640
        batch=32,               # 根据 GPU 显存调整（RTX 3060 可跑 32）
        name='cf_player_detect',
        cache=True,             # 缓存图像加速训练
        device=0,               # 使用 GPU 0
        workers=8,
        patience=20,            # 早停策略
        hsv_h=0.015,            # 色调增强（帮助区分红/绿）
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,           # 旋转增强（应对斜视角）
        translate=0.1,
        scale=0.5,
        mosaic=1.0,             # Mosaic 增强（提升小目标检测）
        flipud=0.0,
        fliplr=0.5,             # 左右翻转
        save=True,
        exist_ok=False
    )

    print(f"✅ 训练完成！最佳模型路径：{results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
```

### 🔧 训练启动命令（命令行方式）

```bash
yolo detect train \
  data=cf_dataset/dataset.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=32 \
  name=cf_player_detect \
  cache=True \
  device=0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=10 fliplr=0.5 mosaic=1.0
```

---

## ✅ 六、验证与测试

```python
# validate.py
from ultralytics import YOLO

model = YOLO('runs/detect/cf_player_detect/weights/best.pt')
metrics = model.val(data='cf_dataset/dataset.yaml')
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

---

## ✅ 七、实时推理（用于游戏辅助/分析）

```python
# infer_game.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/cf_player_detect/weights/best.pt')

# 读取游戏录屏视频
cap = cv2.VideoCapture('cf_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理（置信度阈值可调）
    results = model(frame, conf=0.5)

    # 绘制结果
    annotated_frame = results[0].plot()

    # 可在此添加逻辑：若检测到 enemy 且无 ally → 提醒危险
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        if cls_id == 0:  # enemy
            print("⚠️ 发现敌方玩家！")

    cv2.imshow('CF Player Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 八、性能优化建议

| 问题 | 解决方案 |
|------|--------|
| 敌我颜色相近（如夜间模式） | 增加 **HSV 色彩增强**，或使用 **注意力机制模型** |
| 小目标（远距离玩家）漏检 | 启用 `mosaic=1.0`，使用 `yolov8s` 或更高 |
| 过拟合（游戏画面重复） | 增加 `degrees`, `translate`, `scale` 数据增强 |
| 推理速度慢 | 导出为 ONNX/TensorRT，或使用 `yolov8n` |

---

## ✅ 九、模型导出（部署到边缘设备）

```bash
# 导出 ONNX（用于 OpenCV DNN 或 Unity 插件）
yolo export model=runs/detect/cf_player_detect/weights/best.pt format=onnx

# 导出 TensorRT（高性能推理）
yolo export model=best.pt format=engine
```

---

## ✅ 十、应用场景

| 场景 | 说明 |
|------|------|
| 🎮 **游戏智能辅助** | 实时高亮敌人（仅限单机/训练用途） |
| 📊 **赛事数据分析** | 统计选手视野内敌我出现频率 |
| 🤖 **AI Bot 训练** | 为强化学习提供感知输入 |
| 🎥 **视频自动剪辑** | 检测“1vN”高光时刻并剪辑 |

> ⚠️ **法律与道德提醒**：  
> 本系统 **仅限离线分析、学术研究或单机训练使用**，  
> **禁止用于在线对战作弊**，遵守《穿越火线用户协议》。

---

## ✅ 附：数据集划分建议

| 集合 | 数量 | 用途 |
|------|------|------|
| train | 12,000 | 模型训练 |
| val   | 1,500  | 超参调优、早停 |
| test  | 1,500  | 最终性能评估（不参与训练） |


