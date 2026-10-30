---
layout: default
permalink: /blog/knowledge-video/
title: Knowledge Video
description: 用可视化例子和中文讲解，把机器学习知识点讲清楚。
nav: false
---

<div class="tech-grid" aria-hidden="true"></div>

<div class="tech-home tech-cv">
  <header class="tech-cv-header">
    <div class="tech-prompt"><span class="tech-prompt-user">guest@yingtai</span>:<span class="tech-prompt-path">~/blog</span>$ ls knowledge-video/</div>
    <h1 class="tech-name"><span class="tech-gradient-text">Knowledge Video</span></h1>
    <p class="tech-role">{{ page.description }}</p>
  </header>

  <section class="tech-section">
    <div class="kv-library" data-base="{{ '/assets/videos/' | relative_url }}"></div>
  </section>
</div>

<script defer src="{{ '/assets/js/knowledge-video.js' | relative_url | bust_file_cache }}"></script>
