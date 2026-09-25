---
layout: default
permalink: /publications/
title: publications
description: Publications in differential privacy, program synthesis, and machine learning.
nav: true
nav_order: 2
---

<div class="tech-grid" aria-hidden="true"></div>

<div class="tech-home tech-cv">
  <header class="tech-cv-header">
    <div class="tech-prompt"><span class="tech-prompt-user">guest@yingtai</span>:<span class="tech-prompt-path">~</span>$ ls publications/</div>
    <h1 class="tech-name"><span class="tech-gradient-text">Publications</span></h1>
    <p class="tech-role">{{ page.description }}</p>
    <div class="tech-links">
      <a class="tech-btn" href="https://scholar.google.com/citations?user=eCBNsH0AAAAJ&hl=en" target="_blank" rel="noopener"><i class="ai ai-google-scholar" aria-hidden="true"></i><span>Google Scholar</span></a>
    </div>
  </header>

  <section class="tech-section">
    {% include tech_pubs.liquid by_year=true %}
  </section>
</div>
