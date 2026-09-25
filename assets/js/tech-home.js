// Homepage effects: animated node network, typing text and scroll reveal.
(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------------------
  // Node network background
  // ---------------------------------------------------------------------------
  const canvas = document.getElementById("tech-bg");
  if (canvas && canvas.getContext) {
    const ctx = canvas.getContext("2d");
    let width = 0;
    let height = 0;
    let nodes = [];
    let pointer = { x: -9999, y: -9999 };
    let colors = readColors();

    function readColors() {
      const style = getComputedStyle(document.documentElement);
      return {
        node: style.getPropertyValue("--tech-accent").trim() || "#22d3ee",
        line: style.getPropertyValue("--tech-accent-2").trim() || "#a78bfa",
      };
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round(Math.min(90, (width * height) / 16000));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.6 + 0.8,
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      const maxDist = 140;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < maxDist) {
            ctx.globalAlpha = (1 - d / maxDist) * 0.35;
            ctx.strokeStyle = colors.line;
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
        const pd = Math.hypot(a.x - pointer.x, a.y - pointer.y);
        if (pd < 180) {
          ctx.globalAlpha = (1 - pd / 180) * 0.6;
          ctx.strokeStyle = colors.node;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(pointer.x, pointer.y);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = colors.node;
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function step() {
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > width) n.vx *= -1;
        if (n.y < 0 || n.y > height) n.vy *= -1;
      }
      draw();
      if (!document.hidden) requestAnimationFrame(step);
    }

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", (e) => {
      pointer = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener("pointerleave", () => {
      pointer = { x: -9999, y: -9999 };
    });
    // Re-read colors when the light/dark theme is toggled.
    new MutationObserver(() => {
      colors = readColors();
      if (reduceMotion) draw();
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    if (reduceMotion) {
      draw();
    } else {
      requestAnimationFrame(step);
      document.addEventListener("visibilitychange", () => {
        if (!document.hidden) requestAnimationFrame(step);
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Typing effect
  // ---------------------------------------------------------------------------
  const typing = document.querySelector(".tech-typing-text");
  if (typing && !reduceMotion) {
    let phrases = [];
    try {
      phrases = JSON.parse(typing.dataset.phrases || "[]");
    } catch (e) {
      phrases = [];
    }
    if (phrases.length > 1) {
      let p = 0;
      let c = phrases[0].length;
      let deleting = true;
      const tick = () => {
        const phrase = phrases[p];
        typing.textContent = phrase.slice(0, c);
        let delay = deleting ? 35 : 70;
        if (!deleting && c === phrase.length) {
          deleting = true;
          delay = 2200;
        } else if (deleting && c === 0) {
          deleting = false;
          p = (p + 1) % phrases.length;
          delay = 400;
        } else {
          c += deleting ? -1 : 1;
        }
        setTimeout(tick, delay);
      };
      setTimeout(tick, 2600);
    }
  }

  // ---------------------------------------------------------------------------
  // Reveal sections on scroll
  // ---------------------------------------------------------------------------
  const reveals = document.querySelectorAll(".tech-reveal");
  if ("IntersectionObserver" in window && !reduceMotion) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    reveals.forEach((el) => {
      el.classList.add("is-pending");
      io.observe(el);
    });
  }
})();
