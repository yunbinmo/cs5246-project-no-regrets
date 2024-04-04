document.addEventListener('DOMContentLoaded', function() {
    const contentInput = document.getElementById('sensitiveInput');
    const contentFeedback = document.getElementById('contentFeedback');
    const customModal = document.getElementById('customModal');
    const modalMessage = document.getElementById('modalMessage');
    const closeModal = document.getElementsByClassName("close")[0];

    function handleSensitiveWords(input, feedbackElement) {
        feedbackElement.innerHTML = ''; // Clear previous feedback
        const text = input.value;
        const words = text.split(/\s+/);

        words.forEach(word => {
            const span = document.createElement('span');
            span.textContent = word;
            if (word.toLowerCase() === 'sensitive') {
                span.classList.add('underline');
                span.addEventListener('click', function() {
                    modalMessage.innerHTML = 'This word "Sensitive" is considered sensitive.<br>Suggestion: Consider replacing it with a less sensitive term.';
                    customModal.style.display = "block";
                });
            }
            feedbackElement.appendChild(span);

            // Add a space after each word
            feedbackElement.appendChild(document.createTextNode(' '));
        });
    }

    contentInput.addEventListener('input', function() {
        handleSensitiveWords(this, contentFeedback);
    });

    closeModal.addEventListener('click', function() {
        customModal.style.display = "none";
    });

    window.addEventListener('click', function(event) {
        if (event.target == customModal) {
            customModal.style.display = "none";
        }
    });
});
