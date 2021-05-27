---
layout: post
title: '浅谈深度学习中Attention注意力机制'
date: 2019-01-22
author: vincent
tags: 深度学习,词向量,预训练模型,bert,机器学习,注意力，attention
categories: 文本表示
---


最近在做一个法律方面的文本分类项目，用到了谷歌的预训练模型bert，作为一个萌新，直接上这么高大上的模型感觉有点鸭梨山大，并且对以往用过的attention 机制的具体数学细节有些模糊了，再加上最近比较闲，打算好好写写自己的文章，做个总结，于是乎，有了这篇博客，后面将陆续有transformer、bert 等总结，都是自己的一些学习、项目经验，希望自己能不断进步，也能帮到大家。

## What is attention?
![img](https://raw.githubusercontent.com/ciel-zhang/blog/main/assets/post_img/what_is_attention.jpg)
搞深度学习的童鞋，对attention机制应该都不会陌生，其在谷歌的`Attention is all you need `中大放异彩，并衍生出了 后来的tranformer 模型、GPT以及Bert，GPT以及Bert可以说是自然语言处理的一大里程碑，但他们最重要、底层的机制都是attention .

attention 机制： 通俗地讲，就是一种记忆选择机制，能选择出对模型更重要的信息并予以学习。类似能学习到在情感分析中，‘我喜欢你’这句话的重要信息是‘喜欢’,其可以用在任何的序列模型中，如在以RNN 为网络的编码-解码模型等

## why attention?
我们以传统的seq-to-seq 模型作为示例(当然，注意力机制还能用到CNN甚至其他模型中)，将编码器（encoder）的信息映射到一个固定长度的向量中，将该向量输出到解码器(decoder),如下图所示，一般我们会使用GRU或者LSTM等模型来对序列数据进行编码，然后对RNN直接取最后一个t时刻的hidden state作为decoder的输出，这里会有一些问题:
![img](https://raw.githubusercontent.com/ciel-zhang/blog/main/assets/post_img/att.gif)

1. 将最后一个时刻的隐层输出传入到decoder中这种编码方式，无法体现出不同词语对于下一个预测词的重要程度，如情感分析中“我喜欢你”这句话，明显“喜欢”这个词对目标的重要度最大，故应该对“喜欢”这个词关注更多，但原有模型无法根据需要关注输入序列的相关部分。

2. 当输入的序列长度较长时，固定向量维度较大，在计算上耗时、内存更多，模型效果也不好

## How to use attention?
在seq2seq模型中使用attention机制，与原有的seq2seq不同在于：

首先，encoder传送数据到decoder 部分： 不同于原有RNN模型将最后一个时刻t的hidden state 输出到decoder,而是把encoder每个时刻的hidden state都输出到decoder；

然后， 为了达到attention 的记忆选择目的，decoder 对每个 hidden state 求相应权重(如何计算见后续权重计算部分)，然后对所有hidden states做权重求和计算，从而放大权重大的 hidden state,以达到根据需要关注输入序列的相关部分。

一个完整的seq2seq 注意力机制过程如下所示：

1. decoder (with attention) RNN 接收<END>信息和初始解码器的hidden state (h_init)；

2. RNN处理输入，产生输出和新的hidden state vector(h_4),丢弃输出；

3. 使用encoder hidden states 和h_4 计算该时间步的上下文向量（c_4）;

4. 将h_4 和c_4连接成一个向量，经过一个前馈神经网络产生该时间步的输出o_4；

5. 将h_4和o_4输入到下一时刻的RNN ，重复以上步骤

![img](https://raw.githubusercontent.com/ciel-zhang/blog/main/assets/post_img/nmt.jpg)

## Attention的本质思想
![img](https://raw.githubusercontent.com/ciel-zhang/blog/main/assets/post_img/method.jpg)

如上图所示，将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。


