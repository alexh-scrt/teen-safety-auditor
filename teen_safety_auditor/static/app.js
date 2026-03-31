/**
 * app.js — Client-side JavaScript for Teen Safety Auditor
 *
 * Responsibilities:
 *   - Copy audit results JSON to clipboard
 *   - Download audit report via the /api/audit/report endpoint
 *   - Progressive UI polish (tooltips, keyboard shortcuts, etc.)
 */

'use strict';

// ---------------------------------------------------------------------------
// Copy results to clipboard
// ---------------------------------------------------------------------------

/**
 * Extract the structured result data from the results panel and copy it to
 * the clipboard as pretty-printed JSON.
 *
 * Looks for a hidden <script> tag with id="result-data" injected by the
 * partial template, or falls back to the visible text content of the panel.
 */
async function copyResults() {
  const panel = document.getElementById('results-panel');
  if (!panel) return;

  // Try to find the embedded JSON data element first
  const dataEl = panel.querySelector('[data-result-json]');
  let textToCopy;

  if (dataEl) {
    try {
      const parsed = JSON.parse(dataEl.getAttribute('data-result-json'));
      textToCopy = JSON.stringify(parsed, null, 2);
    } catch (_) {
      textToCopy = dataEl.getAttribute('data-result-json') || '';
    }
  } else {
    // Fallback: copy the visible text
    textToCopy = panel.innerText || panel.textContent || '';
  }

  if (!textToCopy.trim()) {
    showToast('Nothing to copy.', 'warning');
    return;
  }

  try {
    await navigator.clipboard.writeText(textToCopy);
    showToast('Copied to clipboard!', 'success');

    // Visual feedback on the button
    const btn = document.getElementById('btn-copy-results');
    if (btn) {
      const original = btn.textContent;
      btn.textContent = '✓ Copied';
      btn.classList.add('text-green-600', 'border-green-300');
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove('text-green-600', 'border-green-300');
      }, 2000);
    }
  } catch (err) {
    // Clipboard API may be blocked in non-secure contexts
    console.error('Clipboard write failed:', err);
    fallbackCopyToClipboard(textToCopy);
  }
}

/**
 * Fallback clipboard copy using a temporary textarea element for browsers
 * or contexts where the Clipboard API is not available.
 *
 * @param {string} text - The text to copy.
 */
function fallbackCopyToClipboard(text) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.cssText = 'position:fixed;top:-9999px;left:-9999px;opacity:0';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    const success = document.execCommand('copy');
    if (success) {
      showToast('Copied to clipboard!', 'success');
    } else {
      showToast('Copy failed — please copy manually.', 'error');
    }
  } catch (err) {
    showToast('Copy not supported in this browser.', 'error');
  } finally {
    document.body.removeChild(textarea);
  }
}

// ---------------------------------------------------------------------------
// Download audit report
// ---------------------------------------------------------------------------

/**
 * Read the current conversation JSON from the textarea, POST it to the
 * /api/audit/report endpoint, and trigger a browser file download with the
 * returned JSON blob.
 *
 * Provides in-button loading feedback and shows toast notifications on
 * success or failure.
 */
async function downloadReport() {
  const convTextarea = document.getElementById('conversation');
  if (!convTextarea) {
    showToast('No conversation data found.', 'error');
    return;
  }

  const conversationRaw = convTextarea.value.trim();
  if (!conversationRaw) {
    showToast('Please enter a conversation before downloading the report.', 'warning');
    return;
  }

  // Validate JSON before sending
  let turns;
  try {
    turns = JSON.parse(conversationRaw);
    if (!Array.isArray(turns)) {
      throw new Error('Conversation must be a JSON array.');
    }
  } catch (err) {
    showToast(`Invalid JSON: ${err.message}`, 'error');
    return;
  }

  const btn = document.getElementById('btn-download-report');
  const originalText = btn ? btn.textContent : '';

  // Button loading state
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Generating report…';
    btn.classList.add('opacity-70', 'cursor-not-allowed');
  }

  try {
    const response = await fetch('/api/audit/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ turns }),
    });

    if (!response.ok) {
      let errMsg = `Server error ${response.status}`;
      try {
        const errData = await response.json();
        errMsg = errData.detail || errMsg;
      } catch (_) { /* ignore */ }
      throw new Error(errMsg);
    }

    // Extract filename from Content-Disposition header
    const disposition = response.headers.get('Content-Disposition') || '';
    const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
    const filename = filenameMatch ? filenameMatch[1] : 'teen_safety_audit_report.json';

    // Download the blob
    const blob = await response.blob();
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href     = url;
    link.download = filename;
    link.style.cssText = 'display:none';
    document.body.appendChild(link);
    link.click();

    // Clean up
    setTimeout(() => {
      URL.revokeObjectURL(url);
      document.body.removeChild(link);
    }, 1000);

    showToast('Report downloaded successfully!', 'success');
  } catch (err) {
    console.error('Report download failed:', err);
    showToast(`Download failed: ${err.message}`, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = originalText;
      btn.classList.remove('opacity-70', 'cursor-not-allowed');
    }
  }
}

// ---------------------------------------------------------------------------
// Toast notification system
// ---------------------------------------------------------------------------

/** @type {ReturnType<typeof setTimeout>|null} */
let _toastTimer = null;

/**
 * Display a temporary toast notification at the bottom-right of the viewport.
 *
 * @param {string} message  - The message to display.
 * @param {'success'|'warning'|'error'|'info'} [type='info'] - Visual style.
 * @param {number} [duration=3000] - Auto-dismiss delay in milliseconds.
 */
function showToast(message, type = 'info', duration = 3000) {
  // Remove any existing toast
  const existing = document.getElementById('app-toast');
  if (existing) existing.remove();
  if (_toastTimer) clearTimeout(_toastTimer);

  const colorMap = {
    success: 'bg-green-600  text-white',
    warning: 'bg-yellow-500 text-white',
    error:   'bg-red-600    text-white',
    info:    'bg-slate-700  text-white',
  };

  const iconMap = {
    success: '✓',
    warning: '⚠',
    error:   '✕',
    info:    'ℹ',
  };

  const toast = document.createElement('div');
  toast.id = 'app-toast';
  toast.setAttribute('role', 'status');
  toast.setAttribute('aria-live', 'polite');
  toast.className = [
    'fixed bottom-5 right-5 z-50 flex items-center gap-2',
    'rounded-lg shadow-lg px-4 py-3 text-sm font-medium',
    'transition-all duration-300 translate-y-2 opacity-0',
    colorMap[type] || colorMap.info,
  ].join(' ');
  toast.innerHTML = `<span class="font-bold text-base leading-none">${iconMap[type] || 'ℹ'}</span>
                     <span>${escapeHtml(message)}</span>`;

  document.body.appendChild(toast);

  // Trigger entrance animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      toast.classList.remove('translate-y-2', 'opacity-0');
      toast.classList.add('translate-y-0', 'opacity-100');
    });
  });

  // Auto-dismiss
  _toastTimer = setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, duration);

  // Click to dismiss
  toast.addEventListener('click', () => {
    clearTimeout(_toastTimer);
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  });
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Escape HTML special characters to prevent XSS when inserting user-supplied
 * strings into innerHTML.
 *
 * @param {string} str - Raw string to escape.
 * @returns {string} HTML-safe string.
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#039;');
}

/**
 * Format an ISO timestamp string to a locale-aware human-readable string.
 *
 * @param {string} isoString - ISO 8601 datetime string.
 * @returns {string} Formatted date/time, or the original string on parse error.
 */
function formatTimestamp(isoString) {
  if (!isoString) return '';
  try {
    const d = new Date(isoString);
    return d.toLocaleString(undefined, {
      year:   'numeric',
      month:  'short',
      day:    'numeric',
      hour:   '2-digit',
      minute: '2-digit',
      second: '2-digit',
      timeZoneName: 'short',
    });
  } catch (_) {
    return isoString;
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcut: Ctrl/Cmd + Enter to submit active form
// ---------------------------------------------------------------------------

document.addEventListener('keydown', function(evt) {
  const isMac  = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const hotkey = isMac ? evt.metaKey : evt.ctrlKey;

  if (hotkey && evt.key === 'Enter') {
    // Find the visible (non-hidden) form
    const singlePanel = document.getElementById('panel-single');
    const convPanel   = document.getElementById('panel-conversation');

    if (singlePanel && !singlePanel.classList.contains('hidden')) {
      const form = document.getElementById('form-single');
      if (form) {
        evt.preventDefault();
        // Trigger HTMX submit
        htmx.trigger(form, 'submit');
      }
    } else if (convPanel && !convPanel.classList.contains('hidden')) {
      const form = document.getElementById('form-conversation');
      if (form) {
        evt.preventDefault();
        htmx.trigger(form, 'submit');
      }
    }
  }
});

// ---------------------------------------------------------------------------
// Auto-format timestamps in the results panel after HTMX swap
// ---------------------------------------------------------------------------

document.body.addEventListener('htmx:afterSwap', function(evt) {
  const target = evt.detail && evt.detail.target;
  if (!target) return;

  // Replace raw ISO timestamps with formatted versions
  const timestampEls = target.querySelectorAll('[data-timestamp]');
  timestampEls.forEach(el => {
    const iso = el.getAttribute('data-timestamp');
    if (iso) el.textContent = formatTimestamp(iso);
  });
});

// ---------------------------------------------------------------------------
// Expose public API on window for inline onclick handlers in templates
// ---------------------------------------------------------------------------

window.copyResults     = copyResults;
window.downloadReport  = downloadReport;
window.showToast       = showToast;
window.formatTimestamp = formatTimestamp;
