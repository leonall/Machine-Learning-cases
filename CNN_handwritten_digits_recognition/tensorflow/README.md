## 训练

```python
python3 train_mnist_model.py --epochs 20
```

## 预测自己写的数字

```python
python3 predict_mnist_model.py --model mnist_cnn_model_10.ckpt
```

**10 轮训练**

MNIST 测试集 test acc: 0.97

```
0.0.png 9
0.1.png 6
0.2.png 9
1.0.png 1
1.1.png 1
1.2.png 1
2.0.png 2
2.1.png 1
2.2.png 1
3.0.png 3
3.1.png 3
3.2.png 9
4.0.png 4
4.1.png 4
4.2.png 4
5.0.png 3
5.1.png 3
5.2.png 5
6.0.png 1
6.1.png 1
6.2.png 1
7.0.png 7
7.1.png 7
7.2.png 7
8.0.png 8
8.1.png 3
8.2.png 1
9.0.png 9
9.1.png 9
9.2.png 1
```

错误率： 14/30

**20 轮训练**

MNIST 测试集 test accuracy 0.986178

```
0.0.png 9
0.1.png 6
0.2.png 9
1.0.png 1
1.1.png 6
1.2.png 1
2.0.png 2
2.1.png 3
2.2.png 2
3.0.png 3
3.1.png 3
3.2.png 3
4.0.png 4
4.1.png 4
4.2.png 4
5.0.png 3
5.1.png 3
5.2.png 5
6.0.png 6
6.1.png 6
6.2.png 6
7.0.png 7
7.1.png 7
7.2.png 7
8.0.png 8
8.1.png 8
8.2.png 9
9.0.png 9
9.1.png 8
9.2.png 9
```

错误率： 9/30

**30 轮训练**

MNIST 测试集 test accuracy 0.990585

```
0.0.png 9
0.1.png 0
0.2.png 9
1.0.png 1
1.1.png 4
1.2.png 1
2.0.png 2
2.1.png 3
2.2.png 2
3.0.png 3
3.1.png 3
3.2.png 3
4.0.png 4
4.1.png 4
4.2.png 4
5.0.png 3
5.1.png 5
5.2.png 5
6.0.png 6
6.1.png 4
6.2.png 1
7.0.png 7
7.1.png 7
7.2.png 7
8.0.png 8
8.1.png 8
8.2.png 4
9.0.png 9
9.1.png 8
9.2.png 1
```

错误率： 10/30

**50 轮训练**

MNIST 测试集 test accuracy 0.992087

```
0.0.png 9
0.1.png 0
0.2.png 9
1.0.png 1
1.1.png 4
1.2.png 1
2.0.png 2
2.1.png 3
2.2.png 2
3.0.png 3
3.1.png 3
3.2.png 3
4.0.png 4
4.1.png 4
4.2.png 4
5.0.png 3
5.1.png 3
5.2.png 5
6.0.png 6
6.1.png 4
6.2.png 1
7.0.png 7
7.1.png 7
7.2.png 7
8.0.png 8
8.1.png 8
8.2.png 4
9.0.png 9
9.1.png 9
9.2.png 1
```

错误率： 10/30
