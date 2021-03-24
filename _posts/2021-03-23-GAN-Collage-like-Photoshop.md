---
layout: post
comments: true
title: "How to easily edit and compose images using GANs like in Photoshop"
date: 2021-03-23
tags: papers cv
thumbnail: "/assets/images/GAN-Collage-like-Photoshop/thumbnail.jpg"
---


> Using StyleGAN latent space regression to analyze and create image collages with GANs.


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

{% include img.html
            src="GAN-Collage-like-Photoshop/poster.jpg"
            alt="Header img"
            class="center header"
%}

<br>

In this post, I will give a brief overview of the recent paper from MIT *[Using latent space regression to analyze and leverage compositionality in GANs](https://arxiv.org/abs/2103.10951)*.

## ❓What?
Given an incomplete image or a collage of images, generate a realistic image.

![img]({{ '/assets/images/GAN-Collage-like-Photoshop/teaser.gif' | relative_url }})
{: style="width: 100%;" class="center"}

<blockquote class="marked">
<p class="marked">
TL;DR: Train StyleGAN latent space regressor, embed crude collage of images, feed the latent vector to StyleGAN to get a realistic image.
</p>  
</blockquote>


## 📌How?
Using latent space regression to analyze and leverage compositionality in GANs.

1. Train a regressor to predict StyleGAN latent code even from incomplete image  
2. Embedd collage and send it to GAN

This paper presents a simple approach – given a fixed pretrained generator (e.g., StyleGAN), they train a regressor network to predict
the latent code from an input image. To teach the regressor to predict the latent code for images w/ missing pixels they mask random patches during training.
Now, given an input collage, the regressor projects it into a reasonable location of the latent space, which then the generator maps onto the image manifold. Such an approach enables more localized editing of individual image parts compared to direct editing in the latent space

### Interesting findings
- Even though our regressor is never trained on unrealistic and incoherent collages, it projects the given image into a reasonable latent code.
- Authors show that the representation of the generator is already compositional in the latent code. Meaning that altering the part of the input image, will result in a change of the regressed latent code in the corresponding location.


## More results

![img]({{ '/assets/images/GAN-Collage-like-Photoshop/fig-14-15.jpg' | relative_url }})
{: style="width: 100%;" class="center"}

![img]({{ '/assets/images/GAN-Collage-like-Photoshop/fig-16.jpg' | relative_url }})
{: style="width: 100%;" class="center"}

![img]({{ '/assets/images/GAN-Collage-like-Photoshop/fig-17.jpg' | relative_url }})
{: style="width: 100%;" class="center"}



### Related work

- **Paint by Word**.
Recently, I wrote a [blopost]({{'2021-03-24/New-DALL-E-Paint-by-Word' | relative_url }}) of the paper ["Paint by Word"](https://arxiv.org/abs/2103.10951). This paper introduces an Image editing method where the user can paint a mask and specify any text description to guide the image generation in the masked region.


<a name="Conclusion"></a>
## ☑️ Conclusions

### ➕ Pros:
- As input, we need only a single example of approximately how we want the generated image to look (can be a collage of different images).
- Requires only one forward pass of the regressor and generator -> fast, unlike iterative optimization approaches that can require up to a minute to reconstruct an image. https://arxiv.org/abs/1911.11544
- Does not require any labeled attributes.

### Applications
- Image inpainting.
- Example-based image editing (incoherent collage -> to a realistic image) 🔥.

### 📎 References:
📝 Arxiv paper: [arxiv.org/abs/2103.10426](https://arxiv.org/abs/2103.10426)         
🧿 Project page: [chail.github.io/latent-composi…](https://chail.github.io/latent-composition/)  
⚒  GitHub: [Code](https://chail.github.io/latent-composition/)  
📔 Colab: [Link](https://colab.research.google.com/drive/1p-L2dPMaqMyr56TYoYmBJhoyIyBJ7lzH?usp=sharing)

🌐 Related blogpost: [New DALL-E? Paint by Word]({{ '2021-03-24/New-DALL-E-Paint-by-Word' | relative_url }})  



-----

Feel free to ask me any questions in the comments below. Feedback is also very appreciated.  

- **Join** my telegram channel not to miss other novel paper reviews like this! <img style="display:inline" src="{{ '/assets/images/telegram.png' | relative_url }}"> [@gradientdude](https://t.me/gradientdude)
- **Follow** me on twitter <img style="display:inline; height:32px" src="{{ '/assets/images/twitter.png' | relative_url }}"> [@artsiom_s](https://twitter.com/artsiom_s)
