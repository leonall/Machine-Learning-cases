# 为你写诗

采用 TensorFlow 中的 LSTM 单元构建 char-rnn 模型，学习五言、七言古诗，通过多轮训练之后，可以写出比较通顺的诗。

## 获取数据

```bash
wget http://tensorflow-1253902462.cosgz.myqcloud.com/rnn_poetry/poetry
```

## 数据清洗

[generate_poetry.py](generate_poetry.py)

## char-rnn 模型

[poetry_model.py](poetry_model.py)

## 训练

```python
python3 train_poetry.py
```

## 预测（写诗）

```python
python3 predict_poetry.py
```

## “佳作”

训练数据经过清洗和处理，是标准五言或七言的诗歌，因此训练之后，写出来的都是五言或者七言的诗歌。懂机器学习的朋友就会明白，RNN只学到了诗歌的一些“皮毛”，难以写出高质量的诗歌。下面列出一些写的比较好的“佳作”。


```
山风一万里，不得见归心。
风露风流色，春阳雪色流。
秋花不相思，山雨不知秋。
莫见青云水，谁知不到身。
```

```
龙人日出白波中，不到天边白石游。
莫道故来无一事，何因此夜不相愁。
```

```
风山白露出烟烟，独把人书入楚人。
一道不堪多一首，更闻天子自成心。
```


```
风生江上色，水断雁行飞。
```
