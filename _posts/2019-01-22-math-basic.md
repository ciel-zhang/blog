---
layout: post
title: '深度学习中的数学基础:从线性代数到数值优化'
date: 2019-01-22
author: vincent
tags: 深度学习,数学
categories: 数学
nav:
  home: '/'
  tags: '/tags.html'
---

## 第一章 线性代数基础

线性代数在机器学习中起着基殿性的作用，为了能将实际中的数据，如文本、字符串等，输入到计算机，通常需要将字符串数值化和向量化。本章将线性代数基础分为
向量、矩阵和向量的导数这三部分(当然，实际运用中，线性代数的精髓远远不止这三方面，但在机器学习中，我们只要暂时掌握这几部分即可)。

### 1.向量

在线性代数中，**标量**是一个实数，而**向量**是指n个实数组成的有序数组,称为n维向量。
如下述n维列向量：  
    
   $$ a =\left[ 1,2,3\right] ^{T} $$

- 向量的模  
   
    $$ \left | a \right | = \sqrt{\sum_{i=1}^{n}a_i^2} $$
    
- 向量的范数     

    在线性代数中，范数是一个表示“长度”概念的函数，为向量空间内的所有向量赋予非零的正长度或大小，常见的范数有$L_1,L_2$  
      
    - $L_1$范数：  
    
        $$ \left | a \right |_1=\sum_{i=1}^{n}\left | a_i \right | $$
        
    - $L_2$范数：    
    
        $$ \left | a_1 \right |_2=\sqrt{\sum_{i=1}^{n}x_1^2}=\sqrt{x^Tx} $$

- 机器学习中的常用向量     
    - one-hot向量：既是一个n维向量，其中只有一维为1，其他都为0，在文本表示中，常用于wordvec的cbow模型和skip-gram模型训练，可参考下篇深度学习
    中的文本表示

### 2.矩阵
一个大小为$m \times n$的**矩阵**是一个m行n列元素排列成的矩形阵列，一个矩阵A从左上角数起的第i行第j列称为第$i,j$项，记为$A_{i,j}$

#### 2.1 矩阵的基本计算
- 点乘与矩阵乘法   
    - 点乘 (两个矩阵的对应元素相乘)   
    
        $$ A\odot B = A_{i,j}B_{i,j} $$     
        
    - 矩阵乘法  (两个矩阵的乘积仅当第一个矩阵A的列数和另一个矩阵B的行数相等时才能定义)   
    
        $$  (A_{mp}B_{pk})_{ij}=\sum_{k=1}^{p} A_{ik} B_{kj}    $$  
           
- 矩阵的向量化是将矩阵表示成一个列向量（按列）

#### 2.2 常见矩阵和矩阵范数
- 常见矩阵
    - 单位矩阵：主对角元素为1，其他为0 ，单位矩阵均为$n \times n$方阵
    - 对角矩阵：主对角元素不为0，其他元素为0，$ diag(a)$ , $diag(a)b=a \odot b$
    - 对称矩阵： $A = A^T$
- 矩阵范数
    - $L_p$ 范数  
    
        $$ \left \| A \right \|_p = \left ( \sum_{i=1}^{m}\sum_{j=1}^{n}\left | a_{ij} \right |^p \right )^{1/p}$$
        
### 3 向量的导数
导数是微积分中重要的基础概念，在深度学习中，常用于梯度下降优化算法中。

#### 3.1 导数定义
对于定义域和值域都是实数域的函数$y=f(x)$,若$f(x)$在点$x_0$的某个领域$\Delta x$内，极限:

$$ f^`(x_0) = \lim_{\Delta x \rightarrow 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x} $$

存在，则称$y=f(x)$在点$x_0$处可导，若其在定义域包含的某区间内每一个点都可到，则说函数$f(x)$ 在该区间内可到，函数$f(x)$的导数$f(x)^`$也记作$\Delta
 _xf(x),\frac{\partial f(x)}{\partial x} ,\frac{\partial }{\partial x} f(x)$ .
 
#### 3.2 向量的导数
- 实数函数$f(x)=f(x_1,\cdots , x_p) \in \mathbb{R} $关于p维列向量x的导数仍为一个p维向量,求导公式如下：        

$$ \frac{\partial f(x)}{\partial x} = \left [ \frac{\partial f(x)}{\partial x_1},\cdots , \frac{\partial f(x)}{\partial x_p}\right ]^T $$

- 向量函数$f(x)=f(x_1,\cdots , x_p) \in \mathbb{R}^q $关于p维列向量x的导数为一个$p \times q$的矩阵，求导公式如下：      

$$ \frac{\partial f(x)}{\partial x_i} = \left [ \frac{\partial f_1(x)}{\partial x_i},\cdots , \frac{\partial f_q(x)}{\partial x_i}\right ] $$

$$ \frac{\partial f(x)}{\partial x} = \left [ \frac{\partial f(x)}{\partial x_1},\cdots , \frac{\partial f(x)}{\partial x_p}\right ]^T \in \mathbb{R}^{p \times q}  $$

#### 3.3 导数法则

- 加减法则较为简单

- 乘法法则   
    - (1) 若$x \in \mathbb{R}^p  , y=f(x) \in \mathbb{R}^q,z=g(x) \in \mathbb{R}^q $,则         

    $$ \frac{\partial y^T z}{\partial x}= \frac{\partial y}{\partial x}z + \frac{\partial z}{\partial x}y$$

    - (2) 若$x \in \mathbb{R}^p  , y=f(x) \in \mathbb{R}^s,z=g(x) \in \mathbb{R}^t A \in \mathbb{R}^{s \times t} 和x无关 $,则    
    
    $$\frac{\partial y^TAz}{\partial x}= \frac{\partial y}{\partial x}Az + \frac{\partial z}{\partial x}A^Ty$$
    
    - (3) 若$x \in \mathbb{R}^p  , y=f(x) \in \mathbb{R},z=g(x) \in \mathbb{R}^p  $,则    
    
    $$\frac{\partial yz}{\partial x}= \frac{\partial y}{\partial x}z^T + y\frac{\partial z}{\partial x}$$
    
- 链式法则  

链式法则是求复合函数导数的一个重要法则，神经网络中梯度优化经常需要用到,熟练掌握尤为必要    
    - (1) 若$x \in \mathbb{R}^p  , y=g(x) \in \mathbb{R}^s,z=f(y) \in \mathbb{R}^t $,则  
        
   $$\frac{\partial z}{\partial x}= \frac{\partial y}{\partial x} \cdot \frac{\partial z}{\partial y}$$
      
   - (2) 若$X \in \mathbb{R}^{p \times q}  , Y=g(X) \in \mathbb{R}^{s \times t},Z=f(Y) \in \mathbb{R} $,则  
        
   $$\frac{\partial z}{\partial X_{ij}}= tr \left(\left(\frac{\partial z}{\partial Y}\right)^T\frac{\partial Y}{\partial X_{ij}}\right) $$      
      
   tr() 称为n阶矩阵的迹，即$n \times n$矩阵对角线上元素之和

   - (3) 若$X \in \mathbb{R}^{p \times q}  , y=g(X) \in \mathbb{R}^{s},z=f(y) \in \mathbb{R} $,则  
        
   $$\frac{\partial z}{\partial X_{ij}}= \left(\frac{\partial z}{\partial y}\right)^T\frac{\partial Y}{\partial X_{ij}} $$

~2018-07-06~

