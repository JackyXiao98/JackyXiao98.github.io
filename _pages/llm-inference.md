---
layout: default
permalink: /blog/llm-inference/
title: LLM Inference
description: 从“为什么解码这么慢”出发，讲 KV 缓存、GQA / MLA 压缩缓存、FlashAttention、量化、投机解码、连续批处理与 PagedAttention、长上下文的缓存管理，以及推理的权衡与前沿。
nav: false
---

<div class="tech-grid" aria-hidden="true"></div>

<div class="tech-home tech-cv">
  <header class="tech-cv-header">
    <div class="tech-prompt"><span class="tech-prompt-user">guest@yingtai</span>:<span class="tech-prompt-path">~/blog</span>$ ls llm-inference/</div>
    <h1 class="tech-name"><span class="tech-gradient-text">LLM Inference</span></h1>
    <p class="tech-role">大模型推理加速 · {{ page.description }}</p>
    <a class="kv-back" href="{{ '/blog/' | relative_url }}"><i class="ti ti-arrow-left" aria-hidden="true"></i> Blog</a>
  </header>

  <section class="tech-section">
    <div class="kv-library" data-base="{{ '/assets/videos/' | relative_url }}" data-series="llm-inference"></div>
  </section>
</div>

<script defer src="{{ '/assets/js/knowledge-video.js' | relative_url | bust_file_cache }}"></script>
