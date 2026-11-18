/* Reproductor asciinema inline reutilizable.
   Requiere window.REC_CAST (eventos) y window.REC_META. */
(function () {
  'use strict';

  var EVENTS = window.REC_CAST || [];
  var META = window.REC_META || {};
  var shell = document.querySelector('.rec-shell');
  var screenEl = document.getElementById('rec-screen');
  var poster = document.getElementById('rec-poster');
  if (!shell || !screenEl || !EVENTS.length) return;

  var COLS = META.width || 104;
  var ROWS = META.height || 34;
  var DURATION = META.duration || (EVENTS.length ? EVENTS[EVENTS.length - 1][0] : 0);

  var COLORS = {
    30: '#1e1e1e', 31: '#ff6b6b', 32: '#14F195', 33: '#FFB020',
    34: '#79c0ff', 35: '#d2a8ff', 36: '#56d4dd', 37: '#e8edf6',
    90: '#7f8ea3', 91: '#ff8787', 92: '#7ee787', 93: '#ffd166',
    94: '#79c0ff', 95: '#d2a8ff', 96: '#56d4dd', 97: '#ffffff'
  };

  var playBtn = document.getElementById('rec-play');
  var restartBtn = document.getElementById('rec-restart');
  var rangeEl = document.getElementById('rec-range');
  var timeEl = document.getElementById('rec-time');
  var speedEl = document.getElementById('rec-speed');
  var fullBtn = document.getElementById('rec-full');
  var startBtn = document.getElementById('rec-start');
  var copyBtn = document.getElementById('rec-copy');

  var cells, cx, cy, fg, bg, bold, reverse;
  var idx = 0;
  var elapsed = 0;
  var playing = false;
  var started = false;
  var raf = null;
  var playStart = 0;
  var playFrom = 0;
  var speed = 1;
  var lastRender = 0;
  var resumeAfterSeek = false;

  function emptyCell() { return { ch: ' ', fg: null, bg: null, bold: false, reverse: false }; }
  function emptyRow() {
    var row = new Array(COLS);
    for (var i = 0; i < COLS; i++) row[i] = emptyCell();
    return row;
  }

  function resetBuffer() {
    cells = new Array(ROWS);
    for (var y = 0; y < ROWS; y++) cells[y] = emptyRow();
    cx = 0; cy = 0; fg = null; bg = null; bold = false; reverse = false; idx = 0;
  }

  function newline() {
    cx = 0; cy++;
    if (cy >= ROWS) { cells.shift(); cells.push(emptyRow()); cy = ROWS - 1; }
  }

  function writeChar(ch) {
    if (cx >= COLS) newline();
    if (cy < 0) cy = 0;
    if (cy >= ROWS) newline();
    cells[cy][cx] = { ch: ch, fg: fg, bg: bg, bold: bold, reverse: reverse };
    cx++;
  }

  function clearScreen() {
    for (var y = 0; y < ROWS; y++) cells[y] = emptyRow();
    cx = 0; cy = 0;
  }

  function eraseLine(mode) {
    var from = mode === 1 ? 0 : Math.max(cx, 0);
    var to = mode === 1 ? Math.min(cx, COLS - 1) : COLS;
    for (var x = from; x < to; x++) cells[cy][x] = emptyCell();
  }

  function color256(n) {
    n = Math.max(0, Math.min(255, n | 0));
    if (n < 16) {
      var base = ['#000000','#800000','#008000','#808000','#000080','#800080','#008080','#c0c0c0',
                  '#808080','#ff0000','#00ff00','#ffff00','#0000ff','#ff00ff','#00ffff','#ffffff'];
      return base[n];
    }
    if (n >= 232) {
      var v = 8 + (n - 232) * 10;
      return 'rgb(' + v + ',' + v + ',' + v + ')';
    }
    n -= 16;
    var r = Math.floor(n / 36), g = Math.floor((n % 36) / 6), b = n % 6;
    var levels = [0, 95, 135, 175, 215, 255];
    return 'rgb(' + levels[r] + ',' + levels[g] + ',' + levels[b] + ')';
  }

  function applySgr(body) {
    var parts = (body === '' ? ['0'] : body.split(';'));
    for (var i = 0; i < parts.length; i++) {
      var n = parseInt(parts[i] || '0', 10);
      if (n === 0) { fg = null; bg = null; bold = false; reverse = false; }
      else if (n === 1) bold = true;
      else if (n === 22) bold = false;
      else if (n === 7) reverse = true;
      else if (n === 27) reverse = false;
      else if (n === 39) fg = null;
      else if (n === 49) bg = null;
      else if (n >= 30 && n <= 37) fg = COLORS[n];
      else if (n >= 90 && n <= 97) fg = COLORS[n];
      else if (n === 38 || n === 48) {
        var targetBg = n === 48;
        if (parts[i + 1] === '5') {
          var c = color256(parseInt(parts[i + 2], 10));
          if (targetBg) bg = c; else fg = c;
          i += 2;
        } else if (parts[i + 1] === '2') {
          var rr = parseInt(parts[i + 2], 10), gg = parseInt(parts[i + 3], 10), bb = parseInt(parts[i + 4], 10);
          var rgb = 'rgb(' + rr + ',' + gg + ',' + bb + ')';
          if (targetBg) bg = rgb; else fg = rgb;
          i += 4;
        }
      }
    }
  }

  function feed(data) {
    for (var i = 0; i < data.length; i++) {
      var ch = data.charAt(i);
      if (ch === '\x1b') {
        if (data.charAt(i + 1) === '[') {
          var j = i + 2;
          while (j < data.length && !/[A-Za-z]/.test(data.charAt(j))) j++;
          if (j >= data.length) break;
          var body = data.slice(i + 2, j);
          var final = data.charAt(j);
          if (final === 'm') applySgr(body);
          else if (final === 'J') {
            var mode = parseInt(body || '0', 10);
            if (mode === 2 || mode === 3) clearScreen();
          } else if (final === 'H') { cx = 0; cy = 0; }
          else if (final === 'K') { eraseLine(parseInt(body || '0', 10)); }
          i = j;
          continue;
        }
        continue;
      }
      if (ch === '\r') { cx = 0; continue; }
      if (ch === '\n') { newline(); continue; }
      if (ch === '\b') { if (cx > 0) cx--; continue; }
      if (ch === '\t') {
        var spaces = 8 - (cx % 8);
        for (var s = 0; s < spaces; s++) writeChar(' ');
        continue;
      }
      if (ch < ' ') continue;
      writeChar(ch);
    }
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function render() {
    var html = '';
    for (var y = 0; y < ROWS; y++) {
      var row = cells[y];
      var line = '';
      var run = '';
      var runStyle = '';
      for (var x = 0; x < COLS; x++) {
        var cell = row[x];
        var f = cell.fg, b = cell.bg;
        if (cell.reverse) { var tmp = f; f = b || '#05080d'; b = tmp || '#d7e0ea'; }
        var styleParts = [];
        if (f) styleParts.push('color:' + f);
        if (b) styleParts.push('background-color:' + b);
        if (cell.bold) styleParts.push('font-weight:700');
        var style = styleParts.join(';');
        if (style !== runStyle) {
          if (run) line += runStyle ? '<span style="' + runStyle + '">' + escapeHtml(run) + '</span>' : escapeHtml(run);
          run = ''; runStyle = style;
        }
        run += cell.ch;
      }
      if (run) line += runStyle ? '<span style="' + runStyle + '">' + escapeHtml(run) + '</span>' : escapeHtml(run);
      html += '<div class="rec-row">' + (line || ' ') + '</div>';
    }
    screenEl.innerHTML = html;
    if (screenEl.scrollHeight > screenEl.clientHeight + 8) screenEl.scrollTop = screenEl.scrollHeight;
  }

  function applyUpTo(t) {
    while (idx < EVENTS.length && EVENTS[idx][0] <= t + 0.0001) {
      feed(EVENTS[idx][2]);
      idx++;
    }
  }

  function fmt(t) {
    t = Math.max(0, t);
    var m = Math.floor(t / 60);
    var s = Math.floor(t % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function updateUI() {
    var pct = DURATION ? Math.min(1, Math.max(0, elapsed / DURATION)) : 0;
    if (rangeEl) rangeEl.value = Math.round(pct * 1000);
    if (timeEl) timeEl.textContent = fmt(elapsed) + ' / ' + fmt(DURATION);
  }

  function showPoster() { if (poster) poster.hidden = false; }
  function hidePoster() { if (poster) poster.hidden = true; }

  function resetTo(t) {
    elapsed = Math.min(DURATION, Math.max(0, t));
    resetBuffer();
    applyUpTo(elapsed);
    render();
    updateUI();
  }

  function pause() {
    if (!playing) return;
    playing = false;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    if (playBtn) { playBtn.textContent = '▶ Reproducir'; playBtn.classList.add('rec-primary'); }
  }

  function tick(now) {
    if (!playing) return;
    var t = playFrom + ((now - playStart) / 1000) * speed;
    if (t >= DURATION) {
      elapsed = DURATION;
      applyUpTo(DURATION);
      render();
      updateUI();
      pause();
      return;
    }
    elapsed = t;
    applyUpTo(t);
    if (now - lastRender >= 33) { render(); lastRender = now; }
    updateUI();
    raf = requestAnimationFrame(tick);
  }

  function play() {
    if (playing) return;
    if (!started || elapsed >= DURATION - 0.02) { resetTo(0); started = true; }
    hidePoster();
    playing = true;
    playFrom = elapsed;
    playStart = performance.now();
    speed = parseFloat(speedEl ? speedEl.value : '1') || 1;
    if (playBtn) { playBtn.textContent = 'Pausar'; playBtn.classList.remove('rec-primary'); }
    raf = requestAnimationFrame(tick);
  }

  function togglePlay() { if (playing) pause(); else play(); }

  function restart() {
    pause();
    resetTo(0);
    started = true;
    hidePoster();
    play();
  }

  function seekFromRange() {
    if (!rangeEl) return;
    var t = (parseFloat(rangeEl.value) / 1000) * DURATION;
    if (playing) { resumeAfterSeek = true; pause(); }
    resetTo(t);
    started = true;
    hidePoster();
  }

  function plainText() {
    var out = '';
    for (var i = 0; i < EVENTS.length; i++) out += EVENTS[i][2];
    return out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '').replace(/\r/g, '\n');
  }

  function copyCommands() {
    var text = (META.commands && META.commands.length)
      ? '# Comandos del video: ' + (META.title || '') + '\n\n' + META.commands.join('\n')
      : plainText();
    function done() {
      if (!copyBtn) return;
      var old = copyBtn.textContent;
      copyBtn.textContent = '✓ Copiado';
      setTimeout(function () { copyBtn.textContent = old; }, 1800);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(function () { fallbackCopy(text, done); });
    } else {
      fallbackCopy(text, done);
    }
  }

  function fallbackCopy(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {}
    document.body.removeChild(ta);
  }

  // Preview estático para el póster.
  resetBuffer();
  applyUpTo(Math.min(4.2, DURATION));
  render();
  elapsed = 0;
  updateUI();
  showPoster();

  if (playBtn) playBtn.addEventListener('click', togglePlay);
  if (startBtn) startBtn.addEventListener('click', play);
  if (restartBtn) restartBtn.addEventListener('click', restart);
  if (copyBtn) copyBtn.addEventListener('click', copyCommands);
  if (rangeEl) {
    rangeEl.addEventListener('input', seekFromRange);
    rangeEl.addEventListener('change', function () {
      if (resumeAfterSeek) { resumeAfterSeek = false; play(); }
    });
  }
  if (speedEl) {
    speedEl.addEventListener('change', function () {
      var newSpeed = parseFloat(this.value) || 1;
      if (playing) { playFrom = elapsed; playStart = performance.now(); }
      speed = newSpeed;
    });
  }
  if (fullBtn) {
    fullBtn.addEventListener('click', function () {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (shell.requestFullscreen) shell.requestFullscreen();
    });
  }
  document.addEventListener('fullscreenchange', function () {
    if (fullBtn) fullBtn.textContent = document.fullscreenElement ? '⤢ Salir' : '⛶';
  });
  screenEl.tabIndex = 0;
  screenEl.addEventListener('keydown', function (e) {
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); togglePlay(); }
  });
})();
