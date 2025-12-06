'use strict';

let isTyping = false;
let stopTyping = false;

document.addEventListener('DOMContentLoaded', () => {
  const predictionForm = document.getElementById('predictionForm');
  const inputField = predictionForm.querySelector('input[name="user_text"]');
  const submitBtn = predictionForm.querySelector('button[type="submit"]');

  const result = document.getElementById('result');

  predictionForm.addEventListener('submit', async (evt) => {
    evt.preventDefault();

    if (isTyping) {
      isTyping = false;
      stopTyping = true;
      submitBtn.innerHTML = 'Submit';
      return;
    }

    stopTyping = false;
    isTyping = true;
    submitBtn.innerHTML = 'Stop';
    submitBtn.disabled = false;

    const originalBtnText = submitBtn.innerHTML;

    result.innerHTML = '';

    const textData = inputField.value;

    submitBtn.innerHTML = 'Thinking...';
    submitBtn.disabled = true;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: textData}),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Server error');
      }

      const html = marked.parse(data.response);
      const plain = html.replace(/<[^>]+>/g, '');

      typeWriter(result, plain, 300, () => {
        result.innerHTML = html;
        isTyping = false;
        submitBtn.innerHTML = 'Submit';
      });

    } catch (error) {
      console.error('Error:', error);
    } finally {
      submitBtn.innerHTML = originalBtnText;
      submitBtn.disabled = false;
    }
  });
});

function typeWriter(element, text, speed, callback) {
  let i = 0;

  function typing() {
    if (stopTyping) {
      isTyping = false;
      return;
    }

    if (i < text.length) {
      element.innerHTML += text.slice(i, i + 100);
      i+=100;
      setTimeout(typing, speed);
    } else if (callback) {
      callback();
    }
  }

  typing();
}