/* 前端逻辑：上传 → 渲染页面 → 提问 → 画证据框。
   只做展示与请求，不含业务判断；所有阈值都在后端。 */

const state = { docId: null, page: null };

const el = (id) => document.getElementById(id);

function showAnswer(text, meta, citations) {
  el('answer').classList.remove('hidden');
  el('answertext').textContent = text;
  el('meta').textContent = meta;
  const box = el('citations');
  box.innerHTML = '';
  citations.forEach((citation) => {
    const item = document.createElement('div');
    item.className = 'citation';
    const img = document.createElement('img');
    img.src = `/api/documents/${state.docId}/regions/${citation.region_id}.crop`;
    img.alt = '证据裁图';
    const label = document.createElement('span');
    label.textContent = `第 ${citation.page + 1} 页 · ${citation.region_id}`;
    item.appendChild(img);
    item.appendChild(label);
    item.addEventListener('click', () => highlight([citation]));
    box.appendChild(item);
  });
  highlight(citations);
}

/** 在页面图上叠加证据框（按图片实际显示尺寸缩放坐标）。 */
function highlight(citations) {
  const overlay = el('overlay');
  overlay.innerHTML = '';
  const image = el('pageimg');
  if (!state.page || !image.clientWidth) return;
  const scale = image.clientWidth / state.page.width;
  citations
    .filter((citation) => citation.page === 0)
    .forEach((citation) => {
      const [x0, y0, x1, y1] = citation.bbox;
      const node = document.createElement('div');
      node.className = 'box';
      node.style.left = `${x0 * scale}px`;
      node.style.top = `${y0 * scale}px`;
      node.style.width = `${(x1 - x0) * scale}px`;
      node.style.height = `${(y1 - y0) * scale}px`;
      overlay.appendChild(node);
    });
}

async function uploadFile(file) {
  el('docinfo').textContent = '解析中…';
  const form = new FormData();
  form.append('file', file);
  const response = await fetch('/api/documents', { method: 'POST', body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    el('docinfo').textContent = `失败：${detail.detail || response.status}`;
    return;
  }
  const data = await response.json();
  state.docId = data.doc_id;
  state.page = data.pages[0];
  el('docinfo').textContent = `${data.filename} · ${data.page_count} 页 · OCR=${data.ocr} · 视觉=${data.visual}`;
  el('pagewrap').classList.remove('hidden');
  el('pageimg').src = `/api/documents/${data.doc_id}/pages/0.png`;
  el('pageimg').onload = () => highlight([]);
  el('overlay').innerHTML = '';
}

async function askQuestion(question) {
  if (!state.docId) {
    el('docinfo').textContent = '请先上传文档';
    return;
  }
  const response = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: state.docId, question }),
  });
  const data = await response.json();
  if (!response.ok) {
    showAnswer(`请求失败：${data.detail || response.status}`, '', []);
    return;
  }
  const meta = `backend=${data.backend} · 置信度 ${data.confidence.toFixed(2)} · ${data.latency_ms.toFixed(1)} ms`;
  if (data.abstained) {
    showAnswer('证据不足，拒绝作答。', meta, data.citations);
  } else {
    showAnswer(data.answer, meta, data.citations);
  }
}

el('pick').addEventListener('click', () => el('file').click());
el('file').addEventListener('change', (event) => {
  if (event.target.files[0]) uploadFile(event.target.files[0]);
});

const dropzone = el('dropzone');
['dragenter', 'dragover'].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add('drag');
  })
);
['dragleave', 'drop'].forEach((type) =>
  dropzone.addEventListener(type, () => dropzone.classList.remove('drag'))
);
dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]);
});

el('askform').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = el('question').value.trim();
  if (question) askQuestion(question);
});
