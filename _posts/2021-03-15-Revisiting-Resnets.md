---
layout: post
comments: true
title: "You don't need EfficientNets. Simple tricks make ResNets better and faster than EfficientNets"
date: 2021-03-15
tags: papers cv
thumbnail: "/assets/images/Revisiting-Resnets/speed_acc_pareto_curve_zoomed-in.png"
---


> Revisiting ResNets: Improved Training and Scaling Strategies. New family of architectures - *ResNet-RS*.

<!--more-->

<style>
blockquote.marked {
    margin: 0 0 0 0;
    padding: 0 0 0 30px;
    border-left: 0;
    background-color: white;
}

p.marked {
    color: rgb(117, 117, 117);
    font-family: fell, "Noto Sans", "Helvetica Neue", Arial, sans-serif;
    font-weight: 300;
    font-size: 24px;

    letter-spacing: -0.009em;
    line-height: 30px;

    margin-top: 1.75em;
    margin-bottom: 2.02em;
    margin-left: 0px;

}

.container {
    position: relative;
    width: 100%;
    height: 0;
    padding-bottom: 56.25%;
}
.video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

</style>


<div style="display:flex">
     <div style="flex:1;padding-right:5px;">
          <figure>
          <img src="{{ '/assets/images/Revisiting-Resnets/speed_acc_pareto.png' | relative_url }}">
          </figure>
     </div>
     <div style="flex:1;padding-left:5px;">
          <figure>
          <img src="{{ '/assets/images/Revisiting-Resnets/speed_acc_pareto_curve_zoomed-in.png' | relative_url }}">
          </figure>
     </div>
</div>


<br>

In this post I will give a brief overview over the recent paper from Google Brain and UC Berkeley *[Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)*.
Authors introduce a new family of ResNet architectures-- **ResNet-RS**.

# 🔥 Main Results
- ResNet-RSs are 1.7x - 2.7x faster than [EfficientNets](https://arxiv.org/abs/1905.11946) on TPUs, while achieving similar or better accuracies on ImageNet.
- In semi-supervised learning scenario (w/ 130M pseudo-labaled images) ResNet-RS achieves 86.2% top-1 ImageNet accuracy, while being 4.7x faster than [EfficientNet-NoisyStudent](https://arxiv.org/abs/1911.04252) -- SoTA results for transfer learning.

# 🃏 Authors take advantage of the following ideas:
1. Convolutions are better optimized for GPUs/TPUs than depthwise convolutions used in [EfficientNets](https://arxiv.org/abs/1905.11946).
2. Simple Scaling Strategy (i.e. Increasing the model dimensions like width, depth and resolution) is the key. Scale model depth in regimes where overfitting can occur:  
      🔸Depth scaling outperforms width scaling for longer epoch regimes.  
      🔸Width scaling outperforms depth scaling for shorter epoch regimes.  
3. Apply weight decay, label smoothing, dropout and [stochastic depth](https://arxiv.org/abs/1603.0938) for regularization.
4. Use [RandAugment](https://export.arxiv.org/abs/1909.13719) instead of [AutoAugment](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html).
5. Adding two common and simple architectural changes: [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507) and [ResNet-D tricks](https://arxiv.org/abs/1812.01187).
6. Decrease weight decay when using more regularization like dropout, augmentations, stochastic depth, etc.


{% include img.html
            src="Revisiting-Resnets/improvements_ablation_table.png"
            alt="Ablation studies on ImageNet"
            caption="Table: Ablation studies on ImageNet"
            style="width: 100%;"
            class="center"
%}

{% include img.html
            src="Revisiting-Resnets/decrease_weight_decay.png"
            alt="img"
            style="width: 100%;"
            class="center"
%}

## ❓How to tune the hyperparameters?
1. Scaling strategies found in small-scale regimes (e.g. on small models or with few training  epochs) fail to generalize to larger models or longer training iterations
2. Run a small subset of models across different scales, for the full training epochs, to gain intuition on which dimensions are the most useful across model scales.
3. Increase Image Resolution lower than [previously recommended](https://arxiv.org/abs/1905.11946). Larger image resolutions often yield diminishing returns.

{% include img.html
            src="Revisiting-Resnets/scaling_plots.png"
            alt="img"
            class="center"
%}

## ⚔️FLOPs vs Latency
While FLOPs provide a hardware-agnostic metric for assessing computational demand, they may not be indicative of actual latency times for training and inference. In custom hardware architectures (e.g. TPUs and GPUs), FLOPs are an especially poor proxy because operations [are often bounded by memory access costs and have different levels of optimization](https://arxiv.org/abs/1704.04760) on modern matrix multiplication units. The [inverted bottlenecks](https://arxiv.org/abs/1801.04381) used in EfficientNets employ depthwise convolutions with large activations and have a small compute to memory ratio (operational intensity) compared to the ResNet’s bottleneck blocks which employ dense convolutions on smaller activations. This makes EfficientNets less efficient 😂 on modern accelerators compared to ResNets. A ResNet-RS model with 1.8x more FLOPs than EfficientNet-B6 is 2.7x faster on a TPUv3.

{% include img.html
            src="Revisiting-Resnets/memory-param_table-512.png"
            alt="img"
            class="center"
%}

## ⚔️ Parameters vs Memory
Although ResNet-RS has 3.8x more parameters and FLOPs than EfficeintNet with the same accuracy, the ResNet-RS model requires 2.3x less memory and runs ~3x faster on TPUs and GPUs.
Parameter count does not necessarily dictate memory consumption during training because memory is often dominated by the size of the activations. And EfficientNets has large activations which cause a larger memory footprint because EfficientNets requires large image resolutions to match the performance of the ResNet-RSs. E.g, to get 84% top-1 ImageNet accuracy, EficientNet needs an input image of 528x528, while ResNet-RS - only 256x256.

{% include img.html
            src="Revisiting-Resnets/img_size_scaling.png"
            alt="img"
            caption="Figure: Scaling properties of ResNets across varying model scales. Error approximately scales as a power law with FLOPs in the lower FLOPs regime but the trend breaks for larger FLOPs. We observe diminishing returns of scaling the image resolutions beyond 320×320, which motivates the slow image resolution scaling. The scaling configurations run are width  multipliers [0.25,0.5,1.0,1.5,2.0],  depths [26,50,101,200,300,350,400] and image resolutions [128,160,224,320,448]."
            class="center"
%}

## More results

{% include img.html
            src="Revisiting-Resnets/supp_table.png"
            alt="img"
            class="center"
%}

<blockquote class="marked">
<p class="marked">
You'd better use ResNets as baselines for your projects from now on.
</p>  
</blockquote>


<a name="Conclusion"></a>
# ☑️ Conclusions:
1. You'd better **use ResNets as baselines** for your projects from now on.
2. Reporting latencies and memory consumption are generally more relevant metrics to compare different architectures, than the number of FLOPs. **FLOPs and parameters are not representative of latency or memory consumption**.
3. **Training methods can be more task-specific than architectures**. E.g., data augmentation is useful for small datasets or when training for many epochs, but the specifics of the augmentation method can be task-dependent (e.g. scale jittering instead of [RandAugment](https://arxiv.org/abs/1909.13719) is better on Kinetics-400 video classification).
4. The **best performing scaling strategy depends on the training regime and whether overfitting is an issue**. When training for 350 epochs on ImageNet, use depth scaling,  whereas  scaling the width is preferable when training for few epochs (e.g. only 10)
5. Future **successful architectures will probably emerge by co-design with hardware**, particularly in resource-tight regimes like mobile phones.


### 🌐 Links
📝 [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)  
🔨 [Tensorflow implementation](https://github.com/tensorflow/models/tree/master/official/vision/beta)  

### 📎 Other references:
- [EfficientNet](https://arxiv.org/abs/1905.11946)   
- [ResNet-D (Bag of Tricks)](https://arxiv.org/abs/1812.01187)  
- [RandAugment](https://arxiv.org/abs/1909.13719)  
- [AutoAugment](https://arxiv.org/abs/1805.09501)   
- [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507)  




-----

Feel free to ask me any questions in the comments below. Feedback is also very appreciated.  

- **Join** my telegram channel for more reviews like this <img style="display:inline" src="{{ '/assets/images/telegram.png' | relative_url }}"> [@gradientdude](https://t.me/gradientdude)
- **Follow** me on twitter <img style="display:inline; height:32px" src="{{ '/assets/images/twitter.png' | relative_url }}"> [@artsiom_s](https://twitter.com/artsiom_s)
