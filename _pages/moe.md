---
layout: default
permalink: /blog/moe/
title: Mixture of Experts
description: 从“参数多、计算少”的稀疏前馈层出发，讲路由与门控、负载均衡与辅助损失、容量与丢词元、专家分工、细粒度与共享专家、训练稳定性，以及专家并行与 MoE 的权衡。
nav: false
---

<div class="tech-grid" aria-hidden="true"></div>

<div class="tech-home tech-cv">
  <header class="tech-cv-header">
    <div class="tech-prompt"><span class="tech-prompt-user">guest@yingtai</span>:<span class="tech-prompt-path">~/blog</span>$ ls moe/</div>
    <h1 class="tech-name"><span class="tech-gradient-text">Mixture of Experts</span></h1>
    <p class="tech-role">混合专家模型 · {{ page.description }}</p>
    <a class="kv-back" href="{{ '/blog/' | relative_url }}"><i class="ti ti-arrow-left" aria-hidden="true"></i> Blog</a>
  </header>

  <section class="tech-section">
    <div class="kv-library" data-base="{{ '/assets/videos/' | relative_url }}" data-series="moe"></div>
  </section>
</div>

<script defer src="{{ '/assets/js/knowledge-video.js' | relative_url | bust_file_cache }}"></script>
