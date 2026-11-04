"""The console's own CSS and client-side javascript, sent to every page.

Kept in one module instead of the page assembly: it is static markup, it is read
almost never, and it is long enough to hide everything else if left inline.
install() adds it exactly once per process, shared across clients.
"""

from nicegui import ui

HEAD_HTML = '''
<style>
/* The vertical tab selector: a full-height strip on the left. Wide enough
   (and padded on the left) so the running bullet before a tab label, which is
   drawn 6px to the left of the label, is never clipped by the strip edge. */
.tab-strip {
    min-width: 12rem;
    padding-left: 18px !important;
    border-right: 1px solid rgba(255, 255, 255, 0.18);
}
/* A thin rule above the first macOS tab splits the strip into OS groups. */
.tab-strip .q-tab.tab-group-start {
    position: relative;
    margin-top: 10px;
}
.tab-strip .q-tab.tab-group-start::before {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    top: -6px;
    height: 1px;
    background: rgba(255, 255, 255, 0.18);
}
/* Green bullet right before the label of every tab whose service is RUNNING. */
.q-tab.running-bullet .q-tab__content {
    position: relative;
}
.q-tab.running-bullet .q-tab__content::before {
    content: '';
    position: absolute;
    right: 100%;
    margin-right: 6px;
    top: 50%;
    transform: translateY(-50%);
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 4px rgba(0, 255, 136, 0.7);
}
.service-description {
    font-style: italic;
    font-size: 0.8rem;
    opacity: 0.8;
}
.service-status {
    font-size: 1rem;
    font-weight: 700;
    transition: color 0.3s ease-in-out;
}
.model-name {
    font-family: Consolas, Menlo, "Courier New", monospace;
    font-size: calc(1em + 1pt);
}
.legacy-status {
    font-size: 0.9rem;
    font-weight: 600;
    white-space: pre;
}
/* Let long model names truncate instead of stretching the select (a flex item
   refuses to shrink below its content unless told so). */
.model-select .q-field__native .ellipsis {
    min-width: 0;
}
.custom-log {
    scrollbar-width: thin !important;
    scrollbar-color: #4caf50 #1a1a1a !important;
}
.custom-log::-webkit-scrollbar {
    width: 14px !important;
    height: 14px !important;
}
.custom-log::-webkit-scrollbar-track {
    background: #1a1a1a !important;
    border-radius: 7px !important;
}
.custom-log::-webkit-scrollbar-thumb {
    background: #4caf50 !important;
    border-radius: 7px !important;
    border: 3px solid #1a1a1a !important;
}
.custom-log::-webkit-scrollbar-thumb:hover {
    background: #66bb6a !important;
}
/* Copy icon next to a model name: hidden until the row is hovered. */
.model-copy-icon {
    opacity: 0;
    transition: opacity 0.15s ease-in-out;
}
.model-status-row:hover .model-copy-icon {
    opacity: 0.75;
}
.model-copy-icon:hover {
    opacity: 1;
}
/* Cursor-following tooltip: a .cursor-tip-host shows the text of its hidden
   .cursor-tip-text child in #cursor-tip (see the script below). */
.cursor-tip-host {
    width: fit-content;
    cursor: help;
}
.cursor-tip-plain {
    cursor: default;
}
.cursor-tip-text {
    display: none;
}
#cursor-tip {
    position: fixed;
    left: 0;
    top: 0;
    z-index: 10000;
    max-width: 30rem;
    padding: 8px 10px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 6px;
    background: rgba(28, 28, 28, 0.97);
    color: #e8e8e8;
    font-size: 0.8rem;
    line-height: 1.35;
    white-space: pre-line;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    /* Never let the tooltip take the pointer it follows, or it flickers away. */
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.12s ease-in-out;
}
#cursor-tip.visible {
    opacity: 1;
}
</style>
<script>
(() => {
    const OFFSET_X = 16;
    const OFFSET_Y = 18;
    const MARGIN = 8;
    let tip = null;
    let host = null;

    // Created lazily: <body> does not exist yet while the head runs.
    const getTip = () => {
        if (!tip || !tip.isConnected) {
            tip = document.createElement('div');
            tip.id = 'cursor-tip';
            document.body.appendChild(tip);
        }
        return tip;
    };

    const place = (event) => {
        const box = getTip().getBoundingClientRect();
        let x = event.clientX + OFFSET_X;
        let y = event.clientY + OFFSET_Y;
        // Flip to the other side of the pointer rather than run off-screen.
        if (x + box.width > window.innerWidth - MARGIN) {
            x = event.clientX - OFFSET_X - box.width;
        }
        if (y + box.height > window.innerHeight - MARGIN) {
            y = event.clientY - OFFSET_Y - box.height;
        }
        tip.style.left = Math.max(MARGIN, x) + 'px';
        tip.style.top = Math.max(MARGIN, y) + 'px';
    };

    const hide = () => {
        host = null;
        if (tip) tip.classList.remove('visible');
    };

    document.addEventListener('mouseover', (event) => {
        const found = event.target.closest && event.target.closest('.cursor-tip-host');
        if (!found) {
            if (host) hide();
            return;
        }
        if (found === host) return;
        const source = found.querySelector('.cursor-tip-text');
        const text = source ? source.textContent.trim() : '';
        if (!text) { hide(); return; }
        host = found;
        getTip().textContent = text;
        // Position first, then show: no flash in the top-left corner.
        place(event);
        tip.classList.add('visible');
    });

    document.addEventListener('mousemove', (event) => {
        if (!host) return;
        if (!host.isConnected) { hide(); return; }
        place(event);
    });

    document.addEventListener('mouseout', (event) => {
        if (host && !host.contains(event.relatedTarget)) hide();
    });

    // Scrolling moves the host out from under a pointer that never moved.
    document.addEventListener('scroll', hide, true);
    window.addEventListener('blur', hide);
})();
</script>
'''


def install() -> None:
    """Send the stylesheet and the scripts to the clients.

    shared=True: this markup never depends on the client, so NiceGUI serves it
    once rather than repeating it in every page's head."""
    ui.add_head_html(HEAD_HTML, shared=True)
