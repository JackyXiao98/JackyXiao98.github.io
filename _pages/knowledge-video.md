---
layout: default
permalink: /blog/knowledge-video/
title: Knowledge Video
nav: false
---

<!-- Old address: forward to the series page (keeps #video-id deep links working). -->
<p>页面已移动：<a href="{{ '/blog/' | relative_url }}">返回 Blog</a></p>
<script>
  (function () {
    var id = location.hash.slice(1);
    var base = "{{ '/' | relative_url }}";
    if (!id) return location.replace(base + "blog/");
    fetch(base + "assets/videos/catalog.json")
      .then(function (r) { return r.json(); })
      .then(function (c) {
        var s = (c.series || []).find(function (s) { return s.videos.some(function (v) { return v.id === id; }); });
        location.replace(base + "blog/" + (s ? s.id + "/#" + id : ""));
      })
      .catch(function () { location.replace(base + "blog/"); });
  })();
</script>
