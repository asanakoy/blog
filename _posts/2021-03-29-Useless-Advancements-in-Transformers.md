---
layout: post
comments: true
title: "Most of the Recent Advancements in Transformers are Useless"
date: 2021-03-29
tags: papers nlp transformers
thumbnail: "/assets/images/sp/useless_transformers_results.jpg"
---


> Most of recent Transformer Modifications Fail To Transfer Across Implementations and Applications


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

.emoji {
	width: 1.25em;
	vertical-align: top;
	display: inline-block;
	white-space: nowrap;
	overflow: hidden;
	background: no-repeat 2px 50%;
	background-position-y: calc(50% - 1px);
	background-size: 1.25em 1.25em;
	text-indent: -10em;
	padding: 3px 3px 3px 2px;
	margin: -3px -2px;
}

.tgme_widget_message_reply, .tgme_widget_message_link_preview, .tgme_widget_message_game, .tgme_widget_message_invoice {
	display: block;
	position: relative;
	font-size: 14px;
	line-height: 17px;
	padding: 2px 0 2px 12px;
	margin: 6px 0 -2px;
	overflow: hidden;
}

.link_preview_image, .tgme_widget_message_game_image, .tgme_widget_message_invoice_image {
	display: block;
	margin: 7px 0;
	border-radius: 6px;
	background: no-repeat center;
	background-size: cover;
}

.tgme_widget_message_reply::after, .tgme_widget_message_link_preview::before, .tgme_widget_message_game::before {
	content: '';
	position: absolute;
	background-color: #4CA3E2;
	border-radius: 2px;
	width: 2px;
	left: 0;
	top: 2px;
	bottom: 2px;
}

</style>


<div class="tgme_widget_message_text js-message_text" dir="auto"><a target="_blank" rel="noopener" href="https://telegra.ph/file/cc5b46de93b9583b48699.jpg" onclick="return confirm('Open this link?\n\n'+this.href);">​​</a>

<h1>
<b>Most of the Recent Advancements in Transformers are Useless</b><i class="emoji" style="background-image:url('//tlgr.org/img/emoji/40/F09F98B1.png')"><b>😱</b></i>
</h1>
<i>Google Research<br><br></i>Google study <a target="_blank" rel="noopener" href="https://syncedreview.com/2021/03/03/google-study-shows-transformer-modifications-fail-to-transfer-across-implementations-and-applications/" onclick="return confirm('Open this link?\n\n'+this.href);">shows</a> Transformer Modifications Fail To Transfer Across Implementations and Applications.
<br>
<br>
The researchers began by <b>reimplementing and evaluating a variety of transformer variants</b> on the tasks where they are most commonly applied. As a <b>baseline, they used the original transformer</b> model with two modifications: applying <b>layer normalization before the self-attention</b> and feed-forward blocks instead of after, and using <b>relative attention</b> with shared biases instead of sinusoidal positional embeddings.<b>

<h2>
<i class="emoji" style="background-image:url('//tlgr.org/img/emoji/40/F09F9180.png')">
<b>👀</b>
</i>  
Surprise!
</h2>
</b>Most architecture modifications they looked at <b>do not meaningfully improve performance </b>on downstream NLP tasks - they <b>fail to transfer across implementations and applications</b>. See the table below<i class="emoji" style="background-image:url('//tlgr.org/img/emoji/40/F09F9187.png')"><b>👇</b></i> with results for transfer learning based on <a target="_blank" rel="noopener" href="https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html" onclick="return confirm('Open this link?\n\n'+this.href);">T5</a>, and supervised machine translation on the WMT'14 English-German benchmark.

<figure>
<a target="_blank" rel="noopener" class="tgme_widget_message_link_preview" href="{{ '/assets/images/sp/useless_transformers_results.jpg' | relative_url }}">
  <i class="link_preview_image" style="background-image:url({{'/assets/images/sp/useless_transformers_results.jpg' | relative_url }});padding-top:92.25%"></i>
</a>
</figure>

<h2><i class="emoji" style="background-image:url('//tlgr.org/img/emoji/40/F09F9885.png')"><b>😅</b></i> Simple ideas are always the best, and more compute never hurts!</h2>

Modifications that were proved to improve performance are either
<br>(1) relatively simple (e.g. a change in activation function), or
<br>(2) rely on i<b>ncrease in parameter count or FLOPs</b> (e.g. the <a target="_blank" rel="noopener" href="https://arxiv.org/abs/2101.03961" onclick="return confirm('Open this link?\n\n'+this.href);">Switch Transformer</a> or <a target="_blank" rel="noopener" href="https://arxiv.org/abs/1807.03819" onclick="return confirm('Open this link?\n\n'+this.href);">Universal Transformer</a>).
<br> And this makes total sense to me.<br><br>My take on the reasons for such results is that researchers are often pressured by the <b>urge to publishing new papers every year</b>. <i>This spurs</i> <i>cherry-picking of the results, overstated claims, and spurious architectural modifications</i>.  The performance increase shown in many papers is just a result of overfitting over a specific benchmark or more accurate hyperparameter selection compared to the previous work. And such phenomenon is not only inherent for transformer and NLP papers but for other subfields of Deep Learning research as well.  



<br><br><a target="_blank" rel="noopener" href="https://arxiv.org/abs/2102.11972" onclick="return confirm('Open this link?\n\n'+this.href);"><i class="emoji" style="background-image:url('//tlgr.org/img/emoji/40/F09F939D.png')"><b>📝</b></i> Arxiv paper <br>


</a></div>



-----

Feel free to ask me any questions in the comments below. Feedback is also very appreciated.  

- **Join** my telegram channel not to miss other novel paper reviews like this! <img style="display:inline" src="{{ '/assets/images/telegram.png' | relative_url }}"> [@gradientdude](https://t.me/gradientdude)
- **Follow** me on twitter <img style="display:inline; height:32px" src="{{ '/assets/images/twitter.png' | relative_url }}"> [@artsiom_s](https://twitter.com/artsiom_s)
