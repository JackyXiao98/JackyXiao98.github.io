// Knowledge Video library (public): lists the videos from catalog.json and plays
// <id>/video.mp4 with chapters and <id>/content.html as notes. Route: #<video-id>.
// Markup: <div class="kv-library" data-base="/assets/videos/"></div>
(function () {
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmtTime = (t) => Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
  const typeset = (el) => {
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([el]).catch(() => {});
  };

  async function init(root) {
    const base = root.dataset.base;
    let catalog;
    try {
      catalog = await (await fetch(base + "catalog.json", { cache: "no-cache" })).json();
    } catch (e) {
      root.innerHTML = `<p class="protected-error">加载失败：${esc(e.message)}</p>`;
      return;
    }

    root.addEventListener("click", (e) => {
      const a = e.target.closest("[data-seek]");
      const video = root.querySelector("video");
      if (!a || !video) return;
      e.preventDefault();
      video.currentTime = parseFloat(a.dataset.seek);
      video.play();
      video.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    function listView() {
      const cards = catalog.videos
        .map(
          (v) => `
        <a class="kv-card tech-card" href="#${esc(v.id)}">
          <div class="kv-thumb" style="background-image:url('${base}${esc(v.id)}/thumb.jpg')">
            <span class="kv-play"><i class="ti ti-player-play"></i></span>
            <span class="kv-duration">${esc(v.duration)}</span></div>
          <div class="kv-card-body">
            <h3>${esc(v.title)}</h3>
            <p>${esc(v.summary)}</p>
            <div class="kv-tags">${(v.tags || []).map((t) => `<span>${esc(t)}</span>`).join("")}</div>
          </div>
        </a>`
        )
        .join("");
      root.innerHTML = `<p class="kv-intro">${esc(catalog.intro || "")}</p><div class="kv-grid">${cards}</div>`;
    }

    function detailView(v) {
      const chapters = (v.chapters || [])
        .map((c) => `<li><a href="#" data-seek="${c.t}">${fmtTime(c.t)}</a> ${esc(c.title)}</li>`)
        .join("");
      root.innerHTML = `
        <a class="kv-back" href="#"><i class="ti ti-arrow-left"></i> 全部视频</a>
        <h2 class="kv-title">${esc(v.title)}</h2>
        <video controls playsinline preload="metadata" class="protected-video"
          src="${base}${esc(v.id)}/video.mp4" poster="${base}${esc(v.id)}/thumb.jpg"></video>
        ${chapters ? `<h3>章节</h3><ol class="protected-chapters">${chapters}</ol>` : ""}
        <div class="kv-notes"></div>`;
      window.scrollTo({ top: 0 });
      const notes = root.querySelector(".kv-notes");
      fetch(base + v.id + "/content.html", { cache: "no-cache" })
        .then((r) => (r.ok ? r.text() : ""))
        .then((html) => {
          notes.innerHTML = html;
          typeset(notes);
        })
        .catch(() => {});
    }

    function route() {
      const id = decodeURIComponent(location.hash.slice(1));
      const v = catalog.videos.find((x) => x.id === id);
      v ? detailView(v) : listView();
    }
    window.addEventListener("hashchange", route);
    route();
  }

  document.querySelectorAll(".kv-library").forEach(init);
})();
