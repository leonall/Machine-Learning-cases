### 数据

* 训练数据来自于 MNIST 数据集

训练集 55000 张
验证集 5000 张
测试集 10000 张

* 手写数字是采用画图板绘制，预测时，需要压缩到 28x28 的尺寸

| 类别 | \*.0                             | \*.1                             | \*.2                             |
|------|----------------------------------|----------------------------------|----------------------------------|
| 0    | ![](hand_written_digits/0.0.png)   | ![](hand_written_digits/0.1.png) | ![](hand_written_digits/0.2.png) |
| 1    | ![](hand_written_digits/1.0.png)   | ![](hand_written_digits/1.1.png) | ![](hand_written_digits/1.2.png) |
| 2    | ![](hand_written_digits/2.0.png)   | ![](hand_written_digits/2.1.png) | ![](hand_written_digits/2.2.png) |
| 3    | ![](hand_written_digits/3.0.png)   | ![](hand_written_digits/3.1.png) | ![](hand_written_digits/3.2.png) |
| 4    | ![](hand_written_digits/4.0.png)   | ![](hand_written_digits/4.1.png) | ![](hand_written_digits/4.2.png) |
| 5    | ![](hand_written_digits/5.0.png)   | ![](hand_written_digits/5.1.png) | ![](hand_written_digits/5.2.png) |
| 6    | ![](hand_written_digits/6.0.png)   | ![](hand_written_digits/6.1.png) | ![](hand_written_digits/6.2.png) |
| 7    | ![](hand_written_digits/7.0.png)   | ![](hand_written_digits/7.1.png) | ![](hand_written_digits/7.2.png) |
| 8    | ![](hand_written_digits/8.0.png)   | ![](hand_written_digits/8.1.png) | ![](hand_written_digits/8.2.png) |
| 9    | ![](hand_written_digits/9.0.png)   | ![](hand_written_digits/9.1.png) | ![](hand_written_digits/9.2.png) |

### [Keras 模型](keras)

采用 Kears 搭建 CNN 模型，在训练时，对 MNIST 数据集的数据进行了一定的随机变换（平移、旋转、噪音等），增加了数据的丰富性，提高训练模型的鲁棒性。

**训练**

```python
python3 train_mnist_keras.py
```

**预测手写数字**

```python
python3 predict_mnist_model.py --model 训练保存的模型
```

经过 10~50 轮训练，对这 30 个手写数字识别的错误率为 4/30 ~ 5/30。增加训练轮数，对该结果影响不大；不同训练轮数下，预测错误的地方不同，数字 6 是错误率最高的数字。

### [TensorFlow 模型](tensorflow)

TensorFlow 搭建的模型中，没有图形变化的环节，即没有数据集增强环节，预测结果要差一些。错误率在 9/30 ~ 14~30。

## 讨论

整体来看，通过 MNIST 数据训练的 CNN 模型，对自己手写的 30 个数字预测正确率(最高 86.7%， 最低 53.3%)明显低于对 MNIST 数据的精度（MNIST 数据集，随便训练都有 90% 以上的正确率，而本次实验，正确率均超过 97%）。原因可能有：

* 数据集的差异
    * 这 30 个数字是通过鼠标在绘图板上绘制而成，与实际手写数字存在差异
    * 这 30 个数字的图片与 MNIST 数据的生成方式有的差异
* 学习到的模型的泛化能力不够
