"use strict";

async function copyValueText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch (e) {
    // Local files and HTTP previews may need the legacy clipboard path.
  }

  const focused = document.activeElement;
  const selection = window.getSelection();
  const ranges = selection
    ? Array.from({ length: selection.rangeCount }, (_, i) => selection.getRangeAt(i).cloneRange())
    : [];
  const input = document.createElement("textarea");
  input.value = text;
  input.readOnly = true;
  input.style.cssText = "position:fixed;left:-9999px;top:0";
  document.body.appendChild(input);
  try {
    input.select();
    if (!document.execCommand("copy")) throw new Error("Clipboard unavailable");
  } finally {
    input.remove();
    if (focused && focused.isConnected) focused.focus({ preventScroll: true });
    if (selection) {
      selection.removeAllRanges();
      ranges.forEach((range) => selection.addRange(range));
    }
  }
}

function bindCopyBoxes(root) {
  root.querySelectorAll(".smiles-box:not([data-copy-ready])").forEach((box) => {
    const value = box.textContent;
    box.dataset.copyReady = "";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-value";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy value to clipboard");
    button.setAttribute("aria-live", "polite");
    box.appendChild(button);

    let timer, copying = false;
    box.addEventListener("click", async (event) => {
      // Let dragging across a value select text without replacing the clipboard.
      const selection = window.getSelection();
      if (!button.contains(event.target) && selection && !selection.isCollapsed &&
          box.contains(selection.anchorNode) && box.contains(selection.focusNode)) return;
      if (copying) return;
      copying = true;
      clearTimeout(timer);
      try {
        await copyValueText(value);
        box.dataset.copyState = "copied";
        button.textContent = "Copied";
        button.setAttribute("aria-label", "Copied to clipboard");
      } catch (e) {
        box.dataset.copyState = "failed";
        button.textContent = "Copy failed";
        button.setAttribute("aria-label", "Copy failed. Select the text and copy manually.");
        button.title = "Select the text and copy manually.";
      } finally {
        copying = false;
        timer = setTimeout(() => {
          delete box.dataset.copyState;
          button.textContent = "Copy";
          button.setAttribute("aria-label", "Copy value to clipboard");
          button.removeAttribute("title");
        }, 1800);
      }
    });
  });
}
