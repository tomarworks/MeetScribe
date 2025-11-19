import sys
import os
import json
import docx
import traceback
import socket
import re
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory, abort
import subprocess

# ----------------- Config / Args -----------------
if len(sys.argv) < 3:
    print("Usage: python new_UI.py <AUDIO_FILE> <TRANSCRIPT_TEXT>")
    sys.exit(1)

AUDIO_FILE = os.path.abspath(sys.argv[1])
TRANSCRIPT_TEXT = os.path.abspath(sys.argv[2])
UPLOAD_FOLDER = os.path.dirname(AUDIO_FILE)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
TEXT_EXTS = {".txt", ".docx"}

app = Flask(__name__)

# ----------------- Helpers -----------------
def find_free_port(start=5000):
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1

def safe_join(folder, filename):
    if not filename:
        return None
    return os.path.join(folder, os.path.basename(filename))

def list_meetings():
    """
    Group files by base name and return meetings that have audio.
    """
    entries = {}
    for f in sorted(os.listdir(UPLOAD_FOLDER)):
        if f.startswith("."):
            continue
        path = os.path.join(UPLOAD_FOLDER, f)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(f)
        ext = ext.lower()
        ent = entries.setdefault(base, {"base": base, "audio": None, "transcript": None,
                                        "docx": None, "words_json": None, "speakered_txt": None, "mtime": 0})
        if ext in AUDIO_EXTS:
            ent["audio"] = f
        elif ext == ".txt":
            if base.endswith("_speakered") or f.endswith("_speakered.txt"):
                ent["speakered_txt"] = f
            else:
                ent["transcript"] = f
        elif ext == ".docx":
            ent["docx"] = f
        # accept various json names; we prefer *_words.json
        elif f.endswith("_words.json") or f.endswith("_diarized.json") or f.endswith("_speakered.json"):
            ent["words_json"] = f
        try:
            m = os.path.getmtime(path)
            if m > ent["mtime"]:
                ent["mtime"] = m
        except Exception:
            pass

    meetings = []
    for _, info in entries.items():
        if info["audio"]:
            info["mtime_str"] = datetime.fromtimestamp(info["mtime"]).strftime("%d/%m/%Y %H:%M") if info["mtime"] else ""
            meetings.append(info)
    meetings.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return meetings

def load_txt_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [l.rstrip() for l in f.readlines()]
    except Exception:
        return []

def load_docx_lines(path):
    try:
        d = docx.Document(path)
        return [p.text.strip() for p in d.paragraphs if p.text.strip()]
    except Exception:
        return []

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return None

def audio_duration(path):
    try:
        import wave
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None

def parse_speakered_txt(path):
    lines = load_txt_lines(path)
    segments = []
    for ln in lines:
        if ":" in ln:
            sp, txt = ln.split(":", 1)
            segments.append({"speaker": sp.strip(), "text": txt.strip(), "start": None, "end": None})
        else:
            if segments:
                segments[-1]["text"] += " " + ln.strip()
            else:
                segments.append({"speaker": "Unknown", "text": ln.strip(), "start": None, "end": None})
    return segments

def build_diarized_segments(base_info):
    """
    Priority:
      1) <base>_words.json (list of {word,start,end[,speaker]})
      2) <base>_diarized.json / <base>_speakered.json (segments)
      3) <base>_speakered.txt
      4) <base>.txt (whole transcript as Unknown)
    Returns: list of segments {speaker,start,end,text,words?}
    """
    base = base_info["base"]
    audio_file = base_info["audio"]
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file)
    dur = audio_duration(audio_path)

    # candidates – prefer *_words.json
    candidates = []
    fnames = set(os.listdir(UPLOAD_FOLDER))
    if f"{base}_words.json" in fnames:
        candidates.append(os.path.join(UPLOAD_FOLDER, f"{base}_words.json"))
    if f"{base}_diarized.json" in fnames:
        candidates.append(os.path.join(UPLOAD_FOLDER, f"{base}_diarized.json"))
    if f"{base}_speakered.json" in fnames:
        candidates.append(os.path.join(UPLOAD_FOLDER, f"{base}_speakered.json"))
    if base_info.get("words_json"):
        p = os.path.join(UPLOAD_FOLDER, base_info["words_json"])
        if p not in candidates:
            candidates.insert(0, p)

    # Try JSON first
    for cand in candidates:
        data = load_json(cand)
        if not data:
            continue

        # CASE A: list of word dicts (with or without speaker)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "word" in data[0]:
            # normalize
            for w in data:
                if w.get("start") is not None:
                    w["start"] = float(w["start"])
                if w.get("end") is not None:
                    w["end"] = float(w["end"])
                if "speaker" not in w:
                    w["speaker"] = "Unknown"

            # group contiguous words by speaker
            segments = []
            cur_sp = data[0].get("speaker", "Unknown")
            cur_words = []
            for w in data:
                sp = w.get("speaker", "Unknown")
                if sp != cur_sp and cur_words:
                    seg_start = cur_words[0]["start"]
                    seg_end = cur_words[-1].get("end", cur_words[-1]["start"])
                    seg_text = " ".join([x["word"] for x in cur_words])
                    segments.append({"speaker": cur_sp, "start": seg_start, "end": seg_end, "text": seg_text, "words": cur_words})
                    cur_sp = sp
                    cur_words = [w]
                else:
                    cur_words.append(w)
            if cur_words:
                seg_start = cur_words[0]["start"]
                seg_end = cur_words[-1].get("end", cur_words[-1]["start"])
                seg_text = " ".join([x["word"] for x in cur_words])
                segments.append({"speaker": cur_sp, "start": seg_start, "end": seg_end, "text": seg_text, "words": cur_words})

            return segments

        # CASE B: list of segment dicts
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and ("speaker" in data[0] or "text" in data[0]):
            out = []
            for s in data:
                sp = s.get("speaker") or s.get("label") or "Unknown"
                start = float(s.get("start")) if s.get("start") is not None else None
                end = float(s.get("end")) if s.get("end") is not None else None
                text = s.get("text") or s.get("utterance") or ""
                words = s.get("words")
                # normalize words if present
                if isinstance(words, list):
                    clean = []
                    for w in words:
                        word_text = w.get("word") or w.get("text") or w.get("word_text") or ""
                        ws = w.get("start") or w.get("start_time") or w.get("ts") or w.get("t0")
                        we = w.get("end") or w.get("end_time") or w.get("t1")
                        clean.append({"word": word_text,
                                      "start": float(ws) if ws is not None else None,
                                      "end": float(we) if we is not None else None,
                                      "speaker": sp})
                    words = clean
                    if (start is None or end is None) and words:
                        start = words[0]["start"]
                        end = words[-1]["end"] if words[-1]["end"] is not None else words[-1]["start"]
                out.append({"speaker": sp, "start": start, "end": end, "text": text, "words": words})
            return out

    # Speakered TXT fallback
    if base_info.get("speakered_txt") and os.path.exists(os.path.join(UPLOAD_FOLDER, base_info["speakered_txt"])):
        segments = parse_speakered_txt(os.path.join(UPLOAD_FOLDER, base_info["speakered_txt"]))
        if dur and len(segments):
            per = max(0.5, dur / len(segments))
            for i, seg in enumerate(segments):
                seg["start"] = round(i * per, 2)
                seg["end"] = round(min(dur, (i + 1) * per), 2)
        return segments

    # Plain TXT fallback
    if base_info.get("transcript") and os.path.exists(os.path.join(UPLOAD_FOLDER, base_info["transcript"])):
        lines = load_txt_lines(os.path.join(UPLOAD_FOLDER, base_info["transcript"]))
        text = " ".join(lines)
        if dur:
            return [{"speaker": "Unknown", "start": 0.0, "end": dur, "text": text}]
        else:
            return [{"speaker": "Unknown", "start": 0.0, "end": 0.0, "text": text}]

    return []

# ----------------- HTML / JS Template -----------------
HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Diarized Meeting Dashboard</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 18px; color:#111; }
    table { border-collapse: collapse; width: 100%; max-width: 1100px; }
    th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
    th { background:#f5f5f5; font-weight:700; }
    td.left { text-align:left; }
    .control-panel { margin-top:14px; max-width:1100px; display:flex; justify-content:space-between; align-items:center; }
    .btn { padding:6px 10px; border-radius:6px; border:none; cursor:pointer; }
    .btn.rename { background:#f0c14b; }
    .btn.delete { background:#e55353; color:#fff; }
    .audio { margin-top:12px; width:100%; max-width:1100px; }
    .transcript { margin-top:12px; max-width:1100px; border:1px solid #eee; padding:12px; border-radius:6px; background:#fafafa; text-align:left; line-height:1.7; }
    .segment { padding:8px 6px; margin-bottom:6px; border-radius:6px; cursor:pointer; }
    .segment:hover { background:#fff8e1; }
    .speaker { font-weight:700; margin-right:8px; color:#0d47a1; }
    .word { cursor:pointer; padding:0 2px; display:inline-block; }
    .word:hover { background:#fff29a; }
    .word.active { background:#ffe082; }
    .meta { color:#666; font-size:13px; }
    .msg { margin-top:8px; color:#b00; }
  </style>
</head>
<body>
  <h2>Diarized Meeting Dashboard</h2>

  <table id="meetings-table">
    <thead>
      <tr>
        <th>S.No</th>
        <th>DateTime</th>
        <th>Voice File Name</th>
        <th>Audio</th>
        <th>Transcript (download)</th>
      </tr>
    </thead>
    <tbody id="table-body"></tbody>
  </table>

  <div class="control-panel">
    <div>Selected: <strong id="selected-name">None</strong></div>
    <div>
      <input id="rename-input" placeholder="New name (without extension)" />
      <button class="btn rename" onclick="renameSelected()">Rename</button>
      <button class="btn delete" onclick="deleteSelected()">Delete</button>
    </div>
    <div id="action-msg" class="msg"></div>
  </div>

  <div class="audio">
    <audio id="player" controls preload="metadata" style="width:100%;"></audio>
    <div id="player-info" class="meta"></div>
  </div>

  <div id="transcript" class="transcript"></div>
  <div id="summary-box" class="transcript"></div> <!-- ✅ Minutes of Meeting placeholder -->

<script>
async function fetchFiles(){
  const res = await fetch('/api/files');
  if(!res.ok){ document.getElementById('action-msg').textContent = 'Failed to load files'; return; }
  const j = await res.json();
  renderTable(j.files);
}

function encodeName(n){ return encodeURIComponent(n); }

function renderTable(files){
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  files.forEach((f,i) => {
    const tr = document.createElement('tr');
    tr.dataset.base = f.base;
    tr.dataset.audio = f.audio || '';
    tr.dataset.transcript = f.transcript || '';
    const td0 = document.createElement('td'); td0.textContent = i+1; tr.appendChild(td0);
    const td1 = document.createElement('td'); td1.textContent = f.mtime_str || ''; tr.appendChild(td1);
    const td2 = document.createElement('td'); td2.className='left'; td2.textContent = f.audio || ''; tr.appendChild(td2);
    const td3 = document.createElement('td'); 
    if(f.audio){
      const a = document.createElement('a'); a.href = '#'; a.textContent = '🔊'; a.title='Play';
      a.onclick = (e)=>{ e.preventDefault(); selectRow(tr); playMeeting(f); };
      td3.appendChild(a);
    }
    tr.appendChild(td3);
    const td4 = document.createElement('td');
    if(f.transcript || f.words_json || f.docx || f.speakered_txt){
      const doc = f.transcript || f.words_json || f.docx || f.speakered_txt;
      const a2 = document.createElement('a'); a2.href = '/download/' + encodeName(doc); a2.textContent = '📄'; a2.title='Download transcript';
      td4.appendChild(a2);
    }
    tr.appendChild(td4);

    tr.onclick = ()=>{ selectRow(tr); showPreview(f); };
    tbody.appendChild(tr);
  });
}

let selectedRow = null;
function selectRow(tr){
  if(selectedRow) selectedRow.style.background = '';
  selectedRow = tr;
  selectedRow.style.background = '#eef6ff';
  document.getElementById('selected-name').textContent = tr.dataset.audio || tr.dataset.transcript || '(none)';
  document.getElementById('action-msg').textContent = '';
}

function showPreview(f){
  const ta = document.getElementById('transcript');
  ta.innerHTML = '';
  if(f.preview && f.preview.length){
    f.preview.forEach(line=>{
      const d = document.createElement('div'); d.textContent = line; ta.appendChild(d);
    });
  } else {
    ta.textContent = '(no transcript)';
  }
}

async function playMeeting(f){
  const player = document.getElementById('player');
  if(!f.audio){
    document.getElementById('action-msg').textContent = 'No audio';
    return;
  }
  player.src = '/open/' + encodeName(f.audio);
  player.play().catch(()=>{});
  document.getElementById('player-info').textContent = (f.base || '') + ' • ' + (f.mtime_str || '');
  const resp = await fetch('/api/meeting/' + encodeURIComponent(f.base));
  if(!resp.ok){ document.getElementById('action-msg').textContent = 'Failed to load diarized transcript'; return; }
  const info = await resp.json();
  renderDiarized(info);

  // ✅ Fetch Minutes of Meeting summary
  fetch('/api/summarize/' + encodeURIComponent(f.base))
    .then(r => r.json())
    .then(data => {
      if(data.summary){
        document.getElementById('summary-box').innerHTML =
          '<h4>Minutes of Meeting:</h4><ul>' +
          data.summary.split("\\n").map(line => '<li>' + line + '</li>').join('') +
          '</ul>';
      } else {
        document.getElementById('summary-box').innerText = '(no summary available)';
      }
    })
    .catch(()=>{ document.getElementById('summary-box').innerText = 'Summary failed'; });
}

let wordIndexMaps = [];

function renderDiarized(info){
  const ta = document.getElementById('transcript');
  ta.innerHTML = '';
  wordIndexMaps = [];
  if(!info.segments || !info.segments.length){
    ta.textContent = '(no diarized transcript)';
    return;
  }
  const player = document.getElementById('player');

  info.segments.forEach((seg, idx)=>{
    const wrapper = document.createElement('div');
    wrapper.className = 'segment';
    wrapper.dataset.start = seg.start !== null ? seg.start : 0;

    const sp = document.createElement('span');
    sp.className = 'speaker';
    sp.textContent = (seg.speaker || 'Unknown') + ':';
    wrapper.appendChild(sp);

    let starts = [];
    if(seg.words && seg.words.length){
      seg.words.forEach((w, wi)=>{
        const s = document.createElement('span');
        s.className = 'word';
        s.textContent = w.word + ' ';
        s.dataset.start = w.start;
        s.onclick = (e)=>{
          e.stopPropagation();
          try{ player.currentTime = parseFloat(s.dataset.start); player.play(); }catch(e){}
        };
        wrapper.appendChild(s);
        starts.push(typeof w.start === 'number' ? w.start : parseFloat(w.start));
      });
    } else {
      const txt = document.createElement('span');
      txt.textContent = ' ' + (seg.text || '');
      wrapper.appendChild(txt);
    }

    wrapper.onclick = ()=>{
      try{ player.currentTime = parseFloat(wrapper.dataset.start || 0); player.play(); }catch(e){}
    };

    ta.appendChild(wrapper);
    wordIndexMaps.push(starts);
  });

  const allWordSpans = Array.from(document.querySelectorAll('.word'));
  function clearActive(){ allWordSpans.forEach(el => el.classList.remove('active')); }

  player.ontimeupdate = () => {
    const t = player.currentTime || 0;
    const segments = Array.from(document.querySelectorAll('.segment'));
    for (let si = 0; si < segments.length; si++){
      const starts = wordIndexMaps[si];
      if(!starts || starts.length === 0) continue;
      let lo = 0, hi = starts.length - 1, pos = -1;
      while (lo <= hi){
        const mid = (lo + hi) >> 1;
        if (starts[mid] <= t){
          pos = mid; lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      if (pos >= 0){
        clearActive();
        const segWords = segments[si].querySelectorAll('.word');
        if (segWords[pos]) segWords[pos].classList.add('active');
        break;
      }
    }
  };
}

async function renameSelected(){
  if(!selectedRow){ alert('Select a row'); return; }
  const base = selectedRow.dataset.base;
  const newBase = document.getElementById('rename-input').value.trim();
  if(!newBase){ alert('Enter new base'); return; }
  const resp = await fetch('/api/rename', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ base: base, new_base: newBase })
  });
  const j = await resp.json();
  if(resp.ok && j.status==='success'){ fetchFiles(); document.getElementById('action-msg').textContent='Renamed'; }
  else document.getElementById('action-msg').textContent = 'Rename failed: ' + (j.message || '');
}

async function deleteSelected(){
  if(!selectedRow){ alert('Select a row'); return; }
  if(!confirm('Delete meeting files?')) return;
  const base = selectedRow.dataset.base;
  const resp = await fetch('/api/delete', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ base: base })
  });
  const j = await resp.json();
  if(resp.ok && j.status==='success'){ fetchFiles(); document.getElementById('action-msg').textContent='Deleted'; document.getElementById('transcript').innerHTML=''; document.getElementById('summary-box').innerHTML=''; document.getElementById('player').removeAttribute('src'); }
  else document.getElementById('action-msg').textContent = 'Delete failed: ' + (j.message || '');
}

fetchFiles();
</script>
</body>
</html>
"""


# ----------------- API endpoints -----------------
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/files")
def api_files():
    meetings = list_meetings()
    out = []
    for m in meetings:
        preview = []
        if m.get("transcript"):
            preview = load_txt_lines(os.path.join(UPLOAD_FOLDER, m["transcript"]))[:2]
        elif m.get("docx"):
            preview = load_docx_lines(os.path.join(UPLOAD_FOLDER, m["docx"]))[:2]
        out.append({
            "base": m["base"],
            "audio": m.get("audio"),
            "transcript": m.get("transcript"),
            "docx": m.get("docx"),
            "words_json": m.get("words_json"),
            "speakered_txt": m.get("speakered_txt"),
            "mtime_str": m.get("mtime_str"),
            "preview": preview
        })
    return jsonify({"files": out})

@app.route("/api/meeting/<base>")
def api_meeting(base):
    base = os.path.basename(base)
    meetings = list_meetings()
    info = next((m for m in meetings if m["base"] == base), None)
    if not info:
        return jsonify({"error":"not found"}), 404

    segments = build_diarized_segments(info)
    for seg in segments:
        seg.setdefault("speaker", "Unknown")
        seg.setdefault("start", 0.0)
        seg.setdefault("end", None)
        seg.setdefault("text", seg.get("text", ""))
    return jsonify({"base": base, "audio": info.get("audio"), "segments": segments})

@app.route("/download/<path:filename>")
def download(filename):
    safe = safe_join(UPLOAD_FOLDER, filename)
    if not safe or not os.path.exists(safe):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(safe), as_attachment=True)

@app.route("/open/<path:filename>")
def open_inline(filename):
    safe = safe_join(UPLOAD_FOLDER, filename)
    if not safe or not os.path.exists(safe):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(safe), as_attachment=False)

@app.route("/api/rename", methods=["POST"])
def api_rename():
    try:
        data = request.get_json(force=True)
        base = (data.get("base") or "").strip()
        new_base = (data.get("new_base") or "").strip()
        if not base or not new_base:
            return jsonify({"status":"error","message":"missing base or new_base"}), 400

        # sanitize new_base (allow letters, numbers, dash, underscore, dot)
        safe_new = re.sub(r"[^A-Za-z0-9._-]+", "_", new_base).strip("._-")
        if not safe_new:
            return jsonify({"status":"error","message":"invalid new_base"}), 400

        # Gather files to rename for this base
        candidates = []
        # audio with any known extension
        for ext in AUDIO_EXTS:
            src = os.path.join(UPLOAD_FOLDER, f"{base}{ext}")
            if os.path.exists(src):
                candidates.append((f"{base}{ext}", f"{safe_new}{ext}"))
        # common companions
        possible = [
            (f"{base}.txt", f"{safe_new}.txt"),
            (f"{base}.docx", f"{safe_new}.docx"),
            (f"{base}.json", f"{safe_new}.json"),
            (f"{base}_words.json", f"{safe_new}_words.json"),
            (f"{base}_diarized.json", f"{safe_new}_diarized.json"),
            (f"{base}_speakered.json", f"{safe_new}_speakered.json"),
            (f"{base}_speakered.txt", f"{safe_new}_speakered.txt"),
        ]
        for src_name, dst_name in possible:
            src = os.path.join(UPLOAD_FOLDER, src_name)
            if os.path.exists(src):
                candidates.append((src_name, dst_name))

        if not candidates:
            return jsonify({"status":"error","message":"no files found for base"}), 404

        # Collision check
        collisions = []
        for _, dst_name in candidates:
            dst = os.path.join(UPLOAD_FOLDER, dst_name)
            if os.path.exists(dst):
                collisions.append(dst_name)
        if collisions:
            return jsonify({"status":"error","message":"target names already exist: " + ", ".join(collisions)}), 409

        # Perform renames
        for src_name, dst_name in candidates:
            os.rename(os.path.join(UPLOAD_FOLDER, src_name), os.path.join(UPLOAD_FOLDER, dst_name))

        # Try to keep meetings.json in sync if it exists
        meetings_path = os.path.join(UPLOAD_FOLDER, "meetings.json")
        if os.path.exists(meetings_path):
            try:
                with open(meetings_path, "r", encoding="utf-8") as f:
                    meetings = json.load(f)
                changed = False
                for m in meetings:
                    # update name if it starts with old base
                    if isinstance(m.get("name"), str) and os.path.splitext(m["name"])[0] == base:
                        m["name"] = safe_new
                        changed = True
                    # update file fields if present
                    for key in ("audio", "transcript", "words"):
                        val = m.get(key)
                        if isinstance(val, str) and os.path.splitext(val)[0].startswith(base):
                            # handle _words suffix properly
                            if val.endswith("_words.json") and key == "words":
                                m[key] = f"{safe_new}_words.json"
                            elif val.endswith("_diarized.json"):
                                m[key] = f"{safe_new}_diarized.json"
                            elif val.endswith("_speakered.json"):
                                m[key] = f"{safe_new}_speakered.json"
                            elif val.endswith(".json"):
                                m[key] = f"{safe_new}.json"
                            elif any(val.endswith(ext) for ext in AUDIO_EXTS):
                                _, ext = os.path.splitext(val)
                                m[key] = f"{safe_new}{ext}"
                            elif val.endswith(".txt"):
                                m[key] = f"{safe_new}.txt"
                            elif val.endswith(".docx"):
                                m[key] = f"{safe_new}.docx"
                            changed = True
                if changed:
                    with open(meetings_path, "w", encoding="utf-8") as f:
                        json.dump(meetings, f, indent=2)
            except Exception:
                traceback.print_exc()

        return jsonify({"status":"success"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.get_json(force=True)
    base = data.get("base")
    if not base:
        return jsonify({"status":"error","message":"missing parameters"}), 400
    to_delete = []
    for entry in os.listdir(UPLOAD_FOLDER):
        if not os.path.isfile(os.path.join(UPLOAD_FOLDER, entry)):
            continue
        b, ext = os.path.splitext(entry)
        if b == base or entry.startswith(f"{base}_"):
            to_delete.append(entry)
    if not to_delete:
        return jsonify({"status":"error","message":"base not found"}), 404
    try:
        for f in to_delete:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        return jsonify({"status":"success"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500
      


@app.route("/api/summarize/<base>", methods=["GET"])
def summarize(base):
    try:
        meetings = list_meetings()
        meeting = next((m for m in meetings if m["base"] == base), None)
        if not meeting or not meeting.get("transcript"):
            return jsonify({"error": "Transcript not found"}), 404

        transcript_path = os.path.join(UPLOAD_FOLDER, meeting["transcript"])
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        # Ask Ollama for summary
        prompt = f"Summarize the following meeting transcript into clear, concise bullet-point Minutes of Meeting:\n\n{transcript_text}"

        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        summary = result.stdout.decode("utf-8").strip()

        return jsonify({"summary": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------- Run -----------------
if __name__ == "__main__":
    port = find_free_port(5000)
    print(f"Starting UI on http://127.0.0.1:{port}  (folder: {UPLOAD_FOLDER})")
    app.run(debug=True, host="0.0.0.0", port=port)
