/* Belka — product page behaviour: language switch, menu, reveal, gallery lightbox, copy. */
(function () {
  "use strict";

  var doc = document.documentElement;
  var STORE_KEY = "belka-lang";
  doc.classList.add("js");

  /* ------------------------------------------------------------------ English strings.
     Spanish is the page's own markup; it is read from the DOM at start-up so
     switching back needs no second copy of it here. */
  var EN = {
    "meta.title": "Belka — Scan film negatives with your camera and your screen",
    "meta.desc": "Belka is a free, open-source Linux app for digitising film negatives: your screen is the light source, your camera is controlled over USB, and developing works like Lightroom Classic.",

    "nav.skip": "Skip to content",
    "nav.label": "Main",
    "nav.how": "How it works",
    "nav.features": "Features",
    "nav.gallery": "Gallery",
    "nav.cameras": "Cameras",
    "nav.install": "Install",
    "nav.faq": "FAQ",
    "nav.lang": "Language",
    "nav.menu": "Menu",

    "hero.pill": "v0.4 · Free and open source",
    "hero.t1": "Your camera scans film.",
    "hero.t2": "Your screen is the light.",
    "hero.lead": "Belka turns your computer screen into a calibrated light panel, drives your camera over USB and inverts every negative by film density. Then you develop it with Lightroom Classic-style tools, all in one app.",
    "cta.get": "Get it on GitHub",
    "cta.features": "See features",
    "cta.install": "How to install",
    "hero.m1": "Free",
    "hero.m2": "MIT licence",
    "hero.m3": "Linux",
    "hero.m4": "English and Spanish",
    "hero.wintitle": "Belka — Develop",

    "shot.develop.alt": "Belka in the Develop module: the inverted photo in the centre, with the histogram and adjustment panels on the right.",
    "shot.curve.alt": "The Tone Curve panel with a point curve over the histogram.",
    "shot.crop.alt": "Crop and straighten: the whole strip, rotated, with the crop box inside the frame.",
    "shot.guided.alt": "Guided Upright: two guides drawn along building edges straighten the verticals.",
    "shot.compare.alt": "Before and after with a divider: the photo without its develop adjustments on the left, with them on the right.",
    "shot.library.alt": "The Library module with the grid of frames from the roll, with stars and flags.",
    "shot.capture.alt": "The Capture module: controls for the camera connected over USB, the live view already inverted, and the light panel settings.",
    "shot.light.alt": "The light panel: the format area lit with a calibrated tint and registration marks, and the rest of the screen black.",

    "facts.label": "Belka in numbers",
    "facts.f1": "film profiles: <span class=\"nw\">C-41</span>, <span class=\"nw\">ECN-2</span>, B&amp;W and slide",
    "facts.f2": "camera brands with a control table",
    "facts.n3": "16-bit",
    "facts.f3": "TIFF export with an sRGB profile",
    "facts.f4": "local: no account, no cloud, no telemetry",

    "how.kicker": "How it works",
    "how.title": "Three steps from negative to positive.",
    "how.intro": "No scanner needed: a camera with a macro lens, a copy stand and the screen you already own.",
    "dg.title": "Set-up diagram: the camera on a copy stand points down at the film, which lies on a diffuser above the screen lying flat that lights it.",
    "dg.camera": "Camera + macro",
    "dg.stand": "Copy stand",
    "dg.film": "Film",
    "dg.diffuser": "Diffuser",
    "dg.screen": "Screen lying flat = light",
    "how.s1.t": "Lay the screen flat",
    "how.s1.d": "Open the laptop to 180° or lay a monitor down. Belka lights only the area of your film format, measured in real millimetres, and keeps the rest black to avoid flare.",
    "how.s2.t": "Film on top, camera above",
    "how.s2.d": "Place the strip on a diffuser <span class=\"nw\">5–10 mm</span> above the screen. Mount the camera on a copy stand, square to the film, with a macro lens.",
    "how.s3.t": "Capture and get positives",
    "how.s3.d": "Focus with the magnified live view, shoot from the app, and Belka inverts the negative with your film's profile. Ready to develop.",

    "feat.kicker": "Features",
    "feat.title": "The whole darkroom, in one window.",
    "feat.intro": "From the light to the export, every step is built for film. No more inverting by hand with curves.",
    "f.new": "New in 0.4",
    "f.dust.t": "Dust and scratches, with the screen",
    "f.dust.d": "After each frame, a dust shot: the screen turns the film area off and lights a ring around it. Only dust and scratches glow, and Develop erases them, filling in with the film's own grain. Without that shot, it looks for them in the image and leaves stars and highlights alone.",
    "f.edge.t": "Reads the film edge",
    "f.edge.d": "Belka reads the DX barcode printed next to the sprocket holes: it tells you which film it is, offers its profile in one click and gives each capture the negative's own frame number, such as “21A”.",
    "f.dev.t": "Develop like Lightroom Classic",
    "f.dev.d": "A histogram you drag by zones; Basic, Tone Curve, HSL, Color Grading, Detail, Lens Corrections, Transform and Effects panels; before/after and clipping warnings. Keyboard shortcuts match Lightroom wherever they exist.",
    "f.dev.chips": "<li>Basic</li><li>Tone Curve</li><li>HSL / Color</li><li>Color Grading</li><li>Detail</li><li>Lens Corrections</li><li>Transform</li><li>Effects</li>",
    "f.film.t": "44 profiles, density inversion",
    "f.film.d": "Belka measures the film base at the edge and inverts by density, channel by channel. Profiles for <span class=\"nw\">C-41</span>, <span class=\"nw\">ECN-2</span>, black and white and slide are included, and you can save your own from a real roll.",
    "f.film.bw": "B&amp;W",
    "f.light.t": "Light panel calibrated in mm",
    "f.light.d": "Formats from 35 mm to 4×5 in real millimetres, with registration marks. White light, a tint calibrated against the orange mask, or sequential RGB: three exposures combined, like a scanner.",
    "f.cap.t": "Tethered capture",
    "f.cap.d": "ISO, shutter speed, aperture and focus from the app over USB, with a live view that is already inverted and <span class=\"nw\">1–8×</span> magnification to focus on the grain.",
    "f.flat.t": "Flat-field",
    "f.flat.d": "One shot of the light without film corrects lens vignetting and uneven screen light, so skies don't darken in the corners.",
    "f.auto.t": "Auto tone",
    "f.auto.d": "One click (Ctrl+U) sets exposure, contrast, highlights, shadows, whites and blacks. While capturing, Belka suggests the shutter speed that exposes the base well and sets it on the camera with one click.",
    "f.up.t": "Upright and guided perspective",
    "f.up.d": "Upright Auto, Level, Vertical or Full. Or draw 2 to 4 guides along edges that should be straight and the perspective corrects itself; the guides stay movable afterwards.",
    "f.roll.t": "Non-destructive rolls",
    "f.roll.d": "Each roll is a folder with the original RAW files untouched. History with undo and redo, snapshots, and settings you can copy, paste and sync across frames.",
    "f.lib.t": "Stars, flags and clean-up",
    "f.lib.d": "Rate from 0 to 5 stars, flag picks and rejects, and delete failed captures in one go: they go to the trash, so they can be recovered.",
    "f.exp.t": "16-bit TIFF export",
    "f.exp.d": "16-bit TIFF and JPEG with an sRGB ICC profile, ready to print or share. Or a flat, linear version to keep editing in darktable or RawTherapee.",
    "f.exp.chips": "<li>16-bit TIFF</li><li>sRGB JPEG</li><li>Flat linear</li>",

    "gal.kicker": "Gallery",
    "gal.title": "Take a look inside Belka.",
    "gal.intro": "Open any screenshot to see it full size.",
    "g.develop.t": "Develop",
    "g.develop.d": "A draggable histogram and Lightroom Classic-style panels.",
    "g.curve.t": "Tone Curve",
    "g.curve.d": "Parametric and point curves, on RGB or per channel.",
    "g.crop.t": "Crop and straighten",
    "g.crop.d": "The crop stays inside the frame, with no film border.",
    "g.guided.t": "Guided Upright",
    "g.guided.d": "Draw 2 to 4 guides and the perspective is corrected.",
    "g.compare.t": "Before and after",
    "g.compare.d": "With a draggable divider or side by side.",
    "g.library.t": "Library",
    "g.library.d": "The whole roll at a glance, with stars and flags.",
    "g.capture.t": "Capture",
    "g.capture.d": "Shutter speed, aperture, ISO and focus, with the live view already inverted.",
    "g.light.t": "Light panel",
    "g.light.d": "The format's area, in real millimetres.",

    "cam.kicker": "Cameras",
    "cam.title": "Your camera, driven from the app.",
    "cam.intro": "Belka talks to the camera over USB through libgphoto2, the free camera-control library. Connect, adjust and shoot without touching it, so nothing moves between frames.",
    "cam.import": "Can't control your camera over USB? Import your RAW files (NEF, CR3, ARW, RAF, DNG and more), TIFF or JPEG, and Belka inverts them just the same.",
    "cam.brands.t": "Control tables per brand",
    "cam.brands.d": "Each table gives the USB mode to select on the camera. The full list of models is under Capture → Compatible cameras.",
    "cam.honest": "What each model can do (live view, remote focus) depends on libgphoto2.",

    "inst.kicker": "Install",
    "inst.title": "Three commands and you're scanning.",
    "inst.intro": "For Ubuntu and other Linux distributions. No sudo: the installer creates its own Python environment, the <code>belka</code> command and an icon in Applications.",
    "inst.term": "Terminal",
    "inst.copy": "Copy",
    "inst.copied": "Copied",
    "inst.after": "Then open Belka from Applications or type <code>belka</code>. To remove it: <code>./scripts/uninstall.sh</code> (your rolls stay).",
    "inst.req.label": "Requirements",
    "inst.req.sw": "Software",
    "inst.req.sw1": "Ubuntu 26.04 or another <span class=\"nw\">x86-64</span> Linux distribution",
    "inst.req.sw2": "Python 3.11 or newer, with <code>python3-venv</code>",
    "inst.req.sw3": "Your user in the <code>plugdev</code> group to use the camera",
    "inst.req.hw": "Set-up",
    "inst.req.hw1": "A camera that shoots RAW, with a macro lens",
    "inst.req.hw2": "A copy stand or tripod",
    "inst.req.hw3": "A diffuser: <span class=\"nw\">2–3 mm</span> opal acrylic or tracing paper",
    "inst.req.hw4": "A bright screen: a laptop opened to 180° or a monitor lying flat",


    "faq.kicker": "FAQ",
    "faq.title": "Good to know.",
    "faq.q1": "Why use a screen as the light source?",
    "faq.a1": "<p>Because you already have one, and Belka controls it precisely: it lights only the format, in real millimetres; it can tint the light to cancel the orange mask, or take three exposures in pure red, green and blue, like a scanner. Three things to watch:</p><ul><li><strong>A diffuser <span class=\"nw\">5–10 mm</span> above the screen.</strong> If the film touches the screen, the pixel grid shows up in the photo.</li><li><strong><span class=\"nw\">1/30 s</span> or slower.</strong> Many screens dim with PWM and bands appear at fast shutter speeds.</li><li><strong>Night Light off.</strong> It tints the light and changes with the time of day. Belka warns you and, on GNOME, can pause it while it is open.</li></ul>",
    "faq.q2": "Does it replace a film scanner?",
    "faq.a2": "<p>It can: scanning with a camera is fast, and the resolution comes from your camera and macro lens. Sequential RGB mode separates colour the way a scanner does, and the dust shot finds dust and scratches with the screen itself. What it doesn't have is automatic strip advance: you place each frame by hand.</p>",
    "faq.q3": "Does it work with RAW?",
    "faq.a3": "<p>Yes, and it is the recommended way. Capture RAW from the camera, or import RAW files from almost any brand thanks to LibRaw (NEF, CR2, CR3, ARW, RAF, RW2, ORF, PEF, DNG and more), plus TIFF and JPEG. The inversion works on the sensor's linear data and the original files are never modified.</p>",
    "faq.q4": "How accurate are the film profiles?",
    "faq.a4": "<p>The 44 bundled profiles come from published characteristic curves and are approximate. To refine one for your camera and your light, develop a frame with a good base and a neutral grey and use <strong>Save as film profile</strong>: the profile measures the real slopes and is ready for the rest of the roll.</p>",
    "faq.q5": "Are my photos uploaded anywhere?",
    "faq.a5": "<p>No. Belka runs entirely on your computer: no account, no cloud and no telemetry. Your rolls are ordinary folders on your disk.</p>",
    "faq.q6": "How much does it cost?",
    "faq.a6": "<p>Nothing. Belka is free software under the MIT licence: you can use it, study it, change it and share it. The code is on <a href=\"https://github.com/ROMERUU-dev/belka\">GitHub</a>.</p>",

    "final.title": "Your negatives deserve to see the light again.",
    "final.sub": "Free, open source and made for film.",

    "foot.tag": "Film negative digitisation with your camera, and your screen as the light.",
    "foot.nav": "Project",
    "foot.code": "Code on GitHub",
    "foot.issues": "Report an issue",
    "foot.license": "MIT licence",
    "foot.made": "Made by",
    "foot.mit": "Free software under the MIT licence.",
    "foot.tm": "Lightroom is a trademark of Adobe; camera and film names belong to their owners. Belka is not affiliated with them.",

    "lb.label": "Enlarged screenshot",
    "lb.close": "Close",
    "lb.prev": "Previous",
    "lb.next": "Next"
  };

  /* Spanish strings that only live in script (button states). */
  var ES_EXTRA = { "inst.copied": "Copiado" };

  /* ------------------------------------------------------------------ language */
  var textNodes = Array.prototype.slice.call(document.querySelectorAll("[data-i18n]"));
  var attrNodes = Array.prototype.slice.call(document.querySelectorAll("[data-i18n-attr]"));
  var ES = {};

  textNodes.forEach(function (el) { ES[el.getAttribute("data-i18n")] = el.innerHTML; });
  attrNodes.forEach(function (el) {
    parseAttrs(el).forEach(function (p) { ES[p.key] = el.getAttribute(p.attr) || ""; });
  });
  Object.keys(ES_EXTRA).forEach(function (k) { ES[k] = ES_EXTRA[k]; });

  function parseAttrs(el) {
    return el.getAttribute("data-i18n-attr").split(";").map(function (pair) {
      var i = pair.indexOf(":");
      return { attr: pair.slice(0, i).trim(), key: pair.slice(i + 1).trim() };
    });
  }

  var current = "es";

  function shotUrl(shot, ext) { return "img/" + shot + "-" + current + "." + ext; }

  function t(key) {
    var dict = current === "en" ? EN : ES;
    return Object.prototype.hasOwnProperty.call(dict, key) ? dict[key] : ES[key];
  }

  function applyLang(lang) {
    current = lang === "en" ? "en" : "es";
    doc.lang = current;

    textNodes.forEach(function (el) {
      var v = t(el.getAttribute("data-i18n"));
      if (v == null) return;
      if (el.namespaceURI === "http://www.w3.org/2000/svg" || el.tagName === "TITLE") el.textContent = v;
      else el.innerHTML = v;
    });
    attrNodes.forEach(function (el) {
      parseAttrs(el).forEach(function (p) {
        var v = t(p.key);
        if (v != null) el.setAttribute(p.attr, v);
      });
    });

    /* screenshots in the matching language: WebP in each <source>, PNG in the <img> as the fallback
       (sources come first in the DOM, so an <img> re-selects with its new <source> already in place) */
    document.querySelectorAll("source[data-shot], img[data-shot]").forEach(function (el) {
      var isSource = el.tagName === "SOURCE";
      var attr = isSource ? "srcset" : "src";
      var url = shotUrl(el.getAttribute("data-shot"), isSource ? "webp" : "png");
      if (el.getAttribute(attr) !== url) el.setAttribute(attr, url);
    });

    document.querySelectorAll(".lang-switch button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-lang") === current));
    });

    if (lightbox && lightbox.open) showSlide(index);
    doc.classList.remove("i18n-pending");
  }

  function initialLang() {
    /* ?lang=en / ?lang=es in a shared link wins, then the saved choice, then the browser */
    var q = /[?&]lang=(es|en)\b/.exec(location.search);
    if (q) return q[1];
    var saved = null;
    try { saved = localStorage.getItem(STORE_KEY); } catch (e) { /* storage blocked */ }
    if (saved === "es" || saved === "en") return saved;
    return /^es(-|$)/i.test(navigator.language || "") ? "es" : "en";
  }

  document.querySelectorAll(".lang-switch button").forEach(function (b) {
    b.addEventListener("click", function () {
      var lang = b.getAttribute("data-lang");
      try { localStorage.setItem(STORE_KEY, lang); } catch (e) { /* storage blocked */ }
      applyLang(lang);
    });
  });

  /* ------------------------------------------------------------------ header */
  var header = document.querySelector(".site-header");
  function onScroll() { header.classList.toggle("is-scrolled", window.scrollY > 8); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  var menuBtn = document.querySelector(".menu-btn");
  var menu = document.getElementById("nav-menu");
  function setMenu(open) {
    menuBtn.setAttribute("aria-expanded", String(open));
    menu.classList.toggle("is-open", open);
  }
  menuBtn.addEventListener("click", function () { setMenu(menuBtn.getAttribute("aria-expanded") !== "true"); });
  menu.addEventListener("click", function (e) { if (e.target.closest("a")) setMenu(false); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && menu.classList.contains("is-open")) { setMenu(false); menuBtn.focus(); }
  });
  window.matchMedia("(min-width: 981px)").addEventListener("change", function (m) { if (m.matches) setMenu(false); });

  /* ------------------------------------------------------------------ reveal on scroll */
  var reveals = document.querySelectorAll(".reveal");
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) {
    reveals.forEach(function (el) { el.classList.add("is-in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.08 });
    reveals.forEach(function (el, i) {
      /* a small stagger inside each grid */
      var sib = el.parentElement ? Array.prototype.indexOf.call(el.parentElement.children, el) : 0;
      el.style.transitionDelay = Math.min(sib, 5) * 60 + "ms";
      io.observe(el);
    });
  }

  /* ------------------------------------------------------------------ gallery lightbox */
  var lightbox = document.getElementById("lightbox");
  var shots = Array.prototype.slice.call(document.querySelectorAll(".gallery-grid .shot"));
  var index = 0;
  var opener = null;

  function showSlide(i) {
    index = (i + shots.length) % shots.length;
    var btn = shots[index];
    var img = btn.querySelector("img");
    var cap = btn.closest("figure").querySelector("figcaption");
    var lbImg = lightbox.querySelector(".lb-img");
    var shot = img.getAttribute("data-shot");
    lightbox.querySelector(".lb-src").setAttribute("srcset", shotUrl(shot, "webp"));
    lbImg.setAttribute("src", shotUrl(shot, "png"));
    lbImg.alt = img.getAttribute("alt") || "";
    lightbox.querySelector(".lb-cap").innerHTML = cap ? cap.innerHTML : "";
    lightbox.querySelector(".lb-count").textContent = (index + 1) + " / " + shots.length;
  }

  if (lightbox && typeof lightbox.showModal === "function") {
    shots.forEach(function (btn, i) {
      btn.addEventListener("click", function () {
        opener = btn;
        showSlide(i);
        lightbox.showModal();
        doc.classList.add("lb-open");
        lightbox.querySelector(".lb-close").focus();
      });
    });
    lightbox.querySelector(".lb-close").addEventListener("click", function () { lightbox.close(); });
    lightbox.querySelector(".lb-prev").addEventListener("click", function () { showSlide(index - 1); });
    lightbox.querySelector(".lb-next").addEventListener("click", function () { showSlide(index + 1); });
    lightbox.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { e.preventDefault(); showSlide(index - 1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); showSlide(index + 1); }
    });
    /* a click on the dark area (not the picture or a control) closes */
    lightbox.addEventListener("click", function (e) {
      if (e.target === lightbox || e.target.classList.contains("lb-figure") || e.target.classList.contains("lb-top")) lightbox.close();
    });
    lightbox.addEventListener("close", function () {
      doc.classList.remove("lb-open");
      if (opener) opener.focus();
    });
  } else {
    /* no <dialog>: open the picture itself */
    shots.forEach(function (btn) {
      btn.addEventListener("click", function () { var i = btn.querySelector("img"); window.open(i.currentSrc || i.getAttribute("src"), "_blank", "noopener"); });
    });
  }

  /* ------------------------------------------------------------------ copy install commands */
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    var timer = null;
    btn.addEventListener("click", function () {
      var pre = document.querySelector(btn.getAttribute("data-copy"));
      if (!pre) return;
      var text = Array.prototype.map.call(pre.querySelectorAll("code"), function (c) {
        var clone = c.cloneNode(true);
        clone.querySelectorAll(".prompt").forEach(function (p) { p.remove(); });
        return clone.textContent;
      }).join("\n").trim();
      var label = btn.querySelector("[data-i18n]");
      function done() {
        btn.classList.add("is-done");
        label.textContent = t("inst.copied");
        clearTimeout(timer);
        timer = setTimeout(function () { btn.classList.remove("is-done"); label.innerHTML = t("inst.copy"); }, 1800);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { selectText(pre); });
      } else {
        selectText(pre);
      }
    });
  });

  function selectText(el) {
    var r = document.createRange();
    r.selectNodeContents(el);
    var s = window.getSelection();
    s.removeAllRanges();
    s.addRange(r);
  }

  /* ------------------------------------------------------------------ go */
  applyLang(initialLang());
  doc.classList.add("ready");
})();
