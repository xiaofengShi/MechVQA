"use strict";

const controls = document.querySelector(".example-controls");
const exampleButtons = [...controls.querySelectorAll("button")];
const examples = [...document.querySelectorAll(".example")];
function selectExample(name) {
  for (const example of examples)
    example.hidden = example.id !== `example-${name}`;
  for (const button of exampleButtons)
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.example === name),
    );
}
controls.hidden = false;
selectExample("recognition");
for (const button of exampleButtons) {
  button.addEventListener("click", () => selectExample(button.dataset.example));
}

const copyButton = document.querySelector("#copy-citation");
if (navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    const status = document.querySelector("#copy-status");
    try {
      await navigator.clipboard.writeText(
        document.querySelector("#bibtex").textContent,
      );
      status.textContent = "Citation copied.";
    } catch {
      status.textContent =
        "Copy unavailable. Select the citation text or use Download BibTeX.";
    }
  });
}
