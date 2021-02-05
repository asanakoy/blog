---
layout: post
comments: true
title: "Winning solution for Kaggle challenge: Lyft Motion Prediction for Autonomous Vehicles."
date: 2021-02-05
tags: kaggle cv
thumbnail: "/assets/images/Kaggle-Lyft/thumbnail.png"
---


> Competition 3rd Place Solution: Agents' future motion prediction with CNN + Set Transformer. 

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
</style>

![Header]({{ '/assets/images/Kaggle-Lyft/Kaggle-Lyft-header.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 

<br>

In this post we will talk about the solution of our team for the Kaggle competition: [Lyft Motion Prediction for Autonomous Vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/leaderboard), where we have secured 3rd place and have won a prize of <span style="color: green">$6000</span>.

Team “Stochastic Uplift” members: [me](https://gdude.de/), [Dmytro Poplavskiy](https://www.kaggle.com/dmytropoplavskiy), and [Artsem Zhyvalkouski](https://www.kaggle.com/aruchomu).

If you prefer video over text, I have also explained this solution in [YouTube video](https://youtu.be/3Yz8_x38qbc).

# Contents

1. [Problem Description](#problem)  
2. [Input Data and Evaluation Metric](#input)
3. [Method](#method)
	- [Model](#model)
	- [Ensemble learning](#ensembling)
	- [Training details](#training)
	- [Rasterizer parameters and optimization](#rasterizer)
3. [Experiments](#experiments)
	- [Ablation studies](#Ablations)
	- [Final results](#results)
4. [Conclusion](#conclusion)

<a name="problem"></a>
# Problem Description

This competition was organized by [Lyft Level 5](https://self-driving.lyft.com/level5/).

Lyft used a fleet of cars equipped with LIDARS and cameras to collect a large dataset.
They were driving along the streets of Palo Alto and scanned the environment around the cars.

![Cars]({{ '/assets/images/Kaggle-Lyft/cars.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Lyft autonomous vehicle which was used to collect the data. Source: <a href="https://self-driving.lyft.com/level5/">link</a></font>*

The LIDARs and cameras were used to obtain 3D point clouds which were then processed to detect all the surrounding objects (other cars, pedestrians, cyclists, road signs, and so on). Then Lyft engineers registered objects on the map using GPS and the corresponding 3D coordinates. 

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/av-screen1.jpg' | relative_url }}) 
{: style="width: 80%; margin-bottom: 0px; border-width: 0 0 6px 0; border-style: solid;" class="center"} 
![av-pipeline]({{ '/assets/images/Kaggle-Lyft/av-screen3.jpg' | relative_url }}) 
{: style="width: 80%;" class="center"} 
*<font size="2"> Fig: This is how Lyft collected the dataset: (Left) Images captured by cameras, (Right) Visualization of the Lyft's perception system. Source: <a href="https://self-driving.lyft.com/level5/">link</a></font>*

They record a lot of trips to cover very diverse road situations. The collected dataset ([Houston et al., 2020](https://arxiv.org/abs/2006.14480)) contains more than 1000 hours of driving in total.


As you can see in the Figure above the perception task is pretty much solved and the current Lyft pipeline shows fairly robust results.
However, for an autonomous vehicle (AV) to correctly plan its future actions, it needs to know what to anticipate from other agents nearby (e.g., other cars, pedestrians, and cyclists).
<blockquote class="marked">
<p class="marked">
Future motion prediction along with the AV's route planning are still very challenging problems and are yet to be solved for the general case.
</p>  
</blockquote>


![av-pipeline]({{ '/assets/images/Kaggle-Lyft/lyft-av-pipeline-with-red-box.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Full pipeline of the autonomous vehicle decision-making system. From sensory input to path planning. The task of this competition (motion prediciton) is outlined in red. Source: <a href="https://self-driving.lyft.com/level5/">link</a></font>*

In this competition we had to tackle the motion prediction task.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/task.webp' | relative_url }}) 
{: style="width: 65%;" class="center"} 
*<font size="2"> Fig: Future motion prediction problem. The AV needs to know whether the oncoming vehicle will turn right or go straight. Source: <a href="https://www.reddit.com/r/SelfDrivingCars/comments/l1i31x/motion_prediction_winners_of_lyfts_motion">link</a>.</font>*



<a name="input"></a>
# Input Data and Evaluation Metric

We are given a travel history of the agent and its current location on the map which was collected as we described in the previous section.
For every agent we have from 0 to 50 history snapshots with a time interval of 0.1 seconds.
So our input can be represented as an image with $$ 3 + (50 + 1) * 2 = 105$$ channels. Here first 3 channels is the RGB map. Then we have 50 history time steps and one current.
Every time step is represented by two channels: (1) The mask representing the location of the current agent, and (2) the mask representing all other agents nearby. 

<div style="display:flex">
     <div style="flex:1;padding-right:5px;">
          <img src="{{ '/assets/images/Kaggle-Lyft/motion_dataset_example.gif' | relative_url }}">
     </div>
     <div style="flex:1;padding-left:5px;">
          <img src="{{ '/assets/images/Kaggle-Lyft/motion_dataset_with_captions.png' | relative_url }}">
     </div>
</div>
*<font size="2"> Fig: Example of the agent and its history. In this case, the agent is an autonomous vehicle itself, but the dataset contains similar recordings for other cars as well. Colored lines depict road lanes.</font>*

<blockquote class="marked">
<p class="marked">
The task is to predict the trajectory of the agent for the next 50 time steps in future. 
</p>  
</blockquote>
Kaggle test server allowed to submit 3 hypothesis (proposals) for the future trajectory which were compared with the ground truth trajectory.
The evaluation metric of this competition - negative log-likelihood (NLL) of the ground truth coordinates in the distribution defined by the predicted proposals. ***The lower the better.***
In other words, given the ground truth trajectory $$\text{GT}$$ and $$K$$ predicted trajectory hypotheses $$\text{hypothesis}_k, k=1,...,K$$,
we compute the likelihood of the ground truth trajectory under the mixture of Gaussians with the means equal to the predicted trajectories and the Identity matrix as a covariance. 

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/evaluation_metric2.png' | relative_url }}) 
{: style="width: 86%; margin-bottom: 10px" class="center"} 

For every hypothesis we provide a confidence value $$c^{(k)}$$, such that $$\sum_k c^{(k)} = 1$$.
This metric can be further decomposed into the product of 1-dimensional Gaussians, and we get just a log sum of the exponents. Note that in the evaluation metric we ignore the constant normalizing factors of the Gaussian distributions.



<a name="method"></a>
# Method

<a name="model"></a>
## Model

In practice, it is usually a good idea to optimize directly the target metric if possible. So, for the training loss, we used the same function as the evaluation metric (see image above). 

<blockquote class="marked">
<p class="marked">
What you always should do in any Kaggle competition first - is to start with a simple baseline model.  
</p>  
</blockquote>

Our baseline is a simple CNN with one fully-connected layer which takes an input image with $$C$$ channels and predicts $$3$$ trajectories with the corresponding confidences.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/cnn_baseline.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Baseline CNN model.</font>*

And it turned out that this simple baseline model worked the best for us. It became the core of our solution.
Of course, following the common practice, we experimented with different CNN backbones.
Among the best performing backbones were [Xception41](https://arxiv.org/abs/1610.02357), [Xception65](https://arxiv.org/abs/1610.02357), [Xception71](https://arxiv.org/abs/1610.02357), and [EfficientNet B5](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html).

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/cnn_backbones.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Our baseline CNN models with different backbones.</font>*

Unfortunately, we didn’t have time to train all the models until convergence. Given the limited time and resources, the model with Xception41 as a backbone showed the best performance: validation score of 10.37, which is in the range of 5-6th places at the [public Leaderboard](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/leaderboard). Training for longer (especially the model with Xception71 backbone) is likely to improve the score. We observed that our validation scores had a very strong correlation with the scores on the public test set, but usually were lower on some fixed constant.   


Beyond the models with $$3$$ output trajectories, we also trained a ***model with $$16$$ hypotheses*** which resulted in more diverse predictions.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/cnn_baseline_16mod.png' | relative_url }}) 
{: style="width: 70%;" class="center"} 
*<font size="2"> Fig: Our baseline CNN models with 16 hypotheses.</font>*

For example, both left turn and U-turn are plausible in the situation on the image and the $$16$$-output model predicts these in contrast to the previous model with $$3$$ outputs.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/3mod_vs_16mod.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Comparison of the trajectories predicted by (Left) the model with 3 proposals vs (Right) the model with 16 proposals. The model with 16 outputs generates more diverse trajectories.</font>*

<a name="ensembling"></a>
## Ensemble learning

[Model ensembling](https://en.wikipedia.org/wiki/Ensemble_learning) is a very powerful technique. Ensembles combine multiple hypotheses (generated by a set of weaker models) to form a (hopefully) better hypothesis.
We ensembled 7 different models.
Six models with 3 output trajectories:
- Xception41 backbone (x3 different):
	- batch size 64
	- batch size 128
	- weighted average pooling (similar to average pooling but uses learnable weights instead of fixed uniform weights) 
- Xception65
- Xception71
- EfficientNetB5

And one Xception41 model with 16 output hypotheses.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/ensembling1.png' | relative_url }}) 
{: style="width: 90%;" class="center"} 
*<font size="2"> Fig: Our ensemble of 7 models generates 34 different trajectory hypotheses.</font>*

By running inference on all these models we got in total $$6 \times 3 + 1 \times 16 = 34$$ hypotheses and their corresponding confidences.
But we can submit only 3 final hypotheses for evaluation. 



<blockquote class="marked">
<p class="marked">
One of the biggest challenges of this competition is how to combine multiple hypotheses in order to produce only 3 required for the evaluation.
</p>  
</blockquote>


Obviously, it is beneficial to select $$3$$ very diverse proposals as it increases the chance that one of them will be close to the Ground Truth.
One can see that such a problem statement is closely related to a well known [Non-maximum suppression](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) method widely used in object detection pipelines, where multiple spurious detections have to be filtered.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/nms_boxes.png' | relative_url }}) 
{: style="width: 60%;" class="center"} 
*<font size="2"> Fig: Non-maximum suppression for detections.</font>*

A trivial way to do Non-maximum suppression is a greedy algorithm. We iteratively select the most confident detection while suppressing very similar but less confident ones.

We applied a similar greedy approach to our task as well. 
At every step we selected the most confident trajectory and suppressed very similar ones (those which have very small Euclidean distance between one another).
However, this greedy algorithm didn’t work well for our task and gave worse results than the best Xception41 model with 3 hypotheses.

We tried another approach. We applied K-means clustering with 3 clusters to the pool of 34 hypotheses and used their centroids as the final predictions. However, this didn’t work well either.
Another attempt was to compute a weighted average of the clusters using the confidences as weights. Unfortunately, neither of these methods could outperform the best single model with the Xception41 backbone.
We speculate that the discrete nature of greedy non-maximum suppression and K-means significantly limits the possible solution space. 

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/k_means.png' | relative_url }}) 
{: style="width: 93%;" class="center"} 
*<font size="2"> Fig: Ensembling by clustering the hypotheses.</font>*

To combat this shortcoming we considered an extension of the non-maximum suppression to the continuous case.
And these naturally led us to the [Gaussian Mixture Model (GMM)](https://wiki.aalto.fi/pages/viewpage.action?pageId=151492301).
We fit a mixture of 3 Gaussians to the pool of the hypotheses and used the obtained means of the Gaussians as the final predictions. 

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/gmm.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Ensembling by fitting a Gaussian Mixture Model to the hypotheses.</font>*

The optimized objective is very similar to the traditional Gaussian Mixture model objective. It is the log-likelihood of the input hypotheses.
The only modification to the original Gaussian Mixture Model objective is that every hypothesis is weighted according to its confidence.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/gmm_loss.png' | relative_url }}) 
{: style="width: 90%;" class="center"} 

We can optimize this objective using [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) or [Expectation-Maximization (EM)](https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model) algorithm.
This approach works pretty well and produces a solution better than any single model in the ensemble, but it can easily get stuck in local optima and is very sensitive to the initialization.

To further improve the results we implemented a ***Neural Network model which optimizes this loss and can produce a solution in a single forward pass***.

<blockquote class="marked">
<p class="marked">
The important property which we require of this neural network is to be permutation invariant for input hypotheses, because hypotheses produced by CNN models do not have any particular order. 
</p>  
</blockquote>

In other words, a neural network has to take a set of hypotheses as input and produce 3 final trajectories.
As the model satisfying our needs we selected [Set Transformer](https://arxiv.org/abs/1810.00825).

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/with_transformer.png' | relative_url }}) 
{: style="width: 85%;" class="center"} 
*<font size="2"> Fig: Ensembling by using a Set Transformer.</font>*

**Set Transformer** has an encoder that takes input hypotheses and their confidences and applies two [self-attention](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) blocks to them to produce feature vectors that encode all pairwise relations between the input hypotheses.
Next, the first block of the decoder takes 3 learnable seed vectors which represent the prototypes for our three output trajectories, and attend to the most relevant representations of the input hypotheses from the encoder, next follows another self-attention block after which we produce 3 output trajectories and the corresponding confidences.

![av-pipeline]({{ '/assets/images/Kaggle-Lyft/set_transformer.png' | relative_url }}) 
{: style="width: 100%;" class="center"} 
*<font size="2"> Fig: Architecture of our Set Transformer.</font>*


This Transformer is invariant to the permutation of the input trajectories and it does not utilize [positional encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/).
Essentially, it takes a lot of trajectories as inputs and outputs 3 new ones that describe the input in the best possible way.
Such a model can leverage global statistics of the entire training dataset, whereas Gaussian mixture and clustering do not have a global context and work sample-wise.


<a name="training"></a>
## Training details

In this competition, it was crucial to find a proper learning rate schedule. 
We used SGD with a relatively high learning rate of 0.01, batch size 64 (if not stated otherwise), and a [Cosine Annealing scheduler with warm restarts](https://arxiv.org/pdf/1608.03983.pdf). The first cycle length was 50.000 iterations, and we increased the length of every next cycle by a factor of $$\sqrt{2}$$.
Another important trick was gradient clipping (maximal magnitude was set to 2) which stabilized training, especially at the initial epochs. 

![Training loss]({{ '/assets/images/Kaggle-Lyft/train_loss.png' | relative_url }}) 
{: style="width: 65%;" class="center"} 
*<font size="2"> Fig: Training with Cosine Annealing learning rate scheduler. Every next learning rate cycle loss value significantly drops.</font>*

Significant score improvement was achieved after we allowed training samples to be without history at all (before that they were ignored). Such examples can be often encountered in the real-life test when a previously unseen car approaches the AV and the model has to predict the future motion of that car relying merely on its prior knowledge about other road agents.
Utilizing such examples increased the training complexity and improved the model generalization to validation and test sets.

We trained each model for around a week on a single GPU of Titan XP / GTX 2080Ti level. Longer training may bring even better results.


<a name="rasterizer"></a>
## Rasterizer parameters and optimization
To generate the training images from raw data, we used the rasterizer provided by Lyft in the package [l5kit](https://github.com/lyft/l5kit/).
We used default rasterization parameters and produced images of $$224 \times 224$$ pixels. 

<blockquote class="marked">
<p class="marked">
One of the major speed bottlenecks in the pipeline was image rasterization.
</p>  
</blockquote>

Important detail which improved our training efficiency is using a sparse set of only 4 history time steps instead of maximum allowed 50 steps. We used only 1, 2, 4, and 8 ticks in the past. We did not notice any improvement in the target metric score when we used more history steps.

![Sparse history]({{ '/assets/images/Kaggle-Lyft/sparse_history.png' | relative_url }}) 
{: style="width: 60%;" class="center"} 
*<font size="2"> Fig: To increase the model training speed, we used only a sparse set of history snapshots when generated training images: 1, 2, 4, and 8 ticks in the past.</font>*


The default l5kit rasterizer is pretty slow (even with the sparse set of history steps) and produces around 15 images per second on a single CPU core.
To make training faster we optimized the rasterizer. 
First, we uncompressed [z-arr ](https://zarr.readthedocs.io) files that contained raw data and combined multiple operations like a transformation of points into [numba](https://numba.pydata.org/) optimized functions. 
This gave almost x2 speedup.
Then, we cached the rasterized images to disk as compressed [npz](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) files. And during training we just loaded them from disk instead of costly online rasterization. This resulted in x7 speedup and we could read more than a hundred images per second using a single process.

Another important step was to start using the entire dataset which contains more than 1000 hours of driving which is almost 9 times larger than the subset originally provided at Kaggle.
<blockquote class="marked">
<p class="marked">
The full dataset is huge and it has been taking days to finish a single training epoch on it.
</p>  
</blockquote>
However, the dataset is redundant in the sense that consecutive frames (time difference between them is 0.1 sec) are highly correlated.
To allow the model to see more various road situations in a fixed amount of time we removed highly correlated samples by selecting only every 8th frame for every agent in the dataset (subsampling in time with the step of 0.8 seconds). This significantly reduced the epoch training time as well as the disk space required for caching the dataset (from 10 TB to 1.3 TB).



<a name="Experiments"></a>
# Experiments

<a name="Ablations"></a>
## Ablation studies
We conducted ablation studies on one of the intermediate ensemble (before the models converged).

![Results table 1]({{ '/assets/images/Kaggle-Lyft/table_ablations.png' | relative_url }}) 
{: style="width: 90%;" class="center"} 

Our transformer can also be trained in a supervised fashion, using the same loss which we used for our first level CNN models. 
I.e., transformer predicts three trajectories, and the negative log-likelihood loss (NLL) is computed using these three predictions and the ground truth trajectory. 
To prevent data leak we trained this transformer on a hold-out set from the “full train dataset”.
In this case, the transformer performs on par with the Gaussian mixture (12.06 validation loss). However, the transformer is faster during inference because it requires only a single forward pass while the Gaussian Mixture involves a costly optimization process.

Surprisingly, unsupervised training of the transformer on the hold-out training set which doesn’t require GT trajectories performs better than the supervised (12.00 vs 12.06). 
We speculate that the reason for this is that it is hard for the model to learn to produce diverse outputs with supervised loss when most of the hypotheses are too far from the GT trajectory. In this case, the diversity of the output is highly penalized which hinders the model performance.  

We also tried to pre-train the transformer in an unsupervised way and then finetune using the supervised loss. This brought a small improvement by 0.02 (from 12.00 to 11.98). 

Finally, we ***trained the set transformer directly in the validation set in an unsupervised way*** (i.e. without using GT trajectories). This way we achieved the best validation score of **11.82**.

<a name="results"></a>
## Final results
In this table we present our final results. 

![Results table 1]({{ '/assets/images/Kaggle-Lyft/table_final.png' | relative_url }}) 
{: style="width: 90%;" class="center"} 

The ***Set Transformer model trained on the test set directly*** (without using GT of course) is superior to all other models. However, it requires retraining for every novel piece of data 
and maybe less suitable for production than a Transformer pretrained on a hold-out set.
In contrast, transformer pretrained on some large hold-out set is very quick for inference on new observations but is slightly less accurate.


<a name="Conclusion"></a>
# Conclusion

- A simple CNN regression baseline turned out to be the best model and very hard to beat (as always keep it simple).
- Training for longer with the right parameters on the entire dataset was crucial.
- It was very important to optimize the rasterizer to train faster.
- Ensembling with Set Transformer can be used for model ensembling and it lifted our team to 3rd place.
- Unsupervised training of the model directly on the test data yields the best performance. However, if the test data arrives continuously in time, such a model would need to be re-trained for every new chunk of data.

Solution source code and model weights: [GitHub repo](https://github.com/asanakoy/kaggle-lyft-motion-prediction-av).  
Video explanation of the solution: [video](https://youtu.be/3Yz8_x38qbc).

----- 

Feel free to ask me any questions in the comments below. Feedback is also very appreciated.  

Other ways to contact me:
- My telegram channel: [@gradientdude](https://t.me/gradientdude)
- My twitter: [@artsiom_s](https://twitter.com/artsiom_s)

