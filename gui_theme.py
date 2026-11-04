"""The console's CSS, sent once to every page (the legacy tab adds its own)."""

from nicegui import ui

CSS = '''
<style>
/* Vertical tab strip. The left padding leaves room for the running bullet. */
.tab-strip {
    min-width: 12rem;
    padding-left: 18px !important;
    border-right: 1px solid rgba(255, 255, 255, 0.18);
}
/* Thin rule above the first macOS tab: groups the tabs by OS. */
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
/* Green bullet before the label of each RUNNING service. */
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
/* Copy icon next to a model name: visible only while the row is hovered. */
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
/* Green scrollbars for the log windows. */
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
</style>
'''


def install() -> None:
    """Add the CSS to the head of every page (shared: served once, not per page)."""
    ui.add_head_html(CSS, shared=True)
