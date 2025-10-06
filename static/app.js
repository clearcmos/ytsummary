// State management
let currentVideoData = null;
let conversationHistory = [];

// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

document.getElementById('themeToggle').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
});

// Markdown rendering with syntax highlighting
function renderMarkdown(text) {
    const html = marked.parse(text);
    const div = document.createElement('div');
    div.innerHTML = html;

    // Highlight code blocks
    div.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });

    return div.innerHTML;
}

// Status messages
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('statusMessage');
    statusEl.textContent = message;
    statusEl.className = `status-message status-${type}`;
}

// Summarize video
document.getElementById('summarizeBtn').addEventListener('click', async () => {
    const url = document.getElementById('youtubeUrl').value.trim();

    if (!url) {
        showStatus('Please enter a YouTube URL', 'error');
        return;
    }

    const btn = document.getElementById('summarizeBtn');
    btn.disabled = true;
    showStatus('Downloading subtitles...', 'info');

    try {
        // Download subtitles
        const response = await fetch('/api/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        if (!response.ok) {
            throw new Error('Failed to download subtitles');
        }

        const data = await response.json();
        currentVideoData = data;

        // Update UI
        document.getElementById('videoTitle').textContent = data.title;
        document.getElementById('wordCount').textContent = `${data.word_count} words`;
        document.getElementById('chunkCount').textContent = `${data.chunk_count} chunks`;

        // Show summary section
        document.getElementById('summarySection').classList.remove('hidden');
        document.getElementById('summaryContent').innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Analyzing video transcript...</p>
            </div>
        `;

        showStatus('Generating summary...', 'info');

        // Stream summary
        const summaryEventSource = new EventSource(`/api/summary/${data.video_id}`);
        let summaryText = '';

        summaryEventSource.onmessage = (event) => {
            if (event.data === '[DONE]') {
                summaryEventSource.close();
                showStatus('Summary complete!', 'success');
                document.getElementById('qaSection').classList.remove('hidden');
                return;
            }

            try {
                const chunk = JSON.parse(event.data);
                summaryText = chunk.accumulated;
                document.getElementById('summaryContent').innerHTML = renderMarkdown(summaryText);
            } catch (e) {
                console.error('Parse error:', e);
            }
        };

        summaryEventSource.onerror = () => {
            summaryEventSource.close();
            showStatus('Error generating summary', 'error');
        };

    } catch (error) {
        showStatus(error.message, 'error');
    } finally {
        btn.disabled = false;
    }
});

// Ask question
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();

    if (!question || !currentVideoData) return;

    const askBtn = document.getElementById('askBtn');
    askBtn.disabled = true;

    // Add user message to chat
    const chatContainer = document.getElementById('chatContainer');
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user-message';
    userMessageDiv.innerHTML = `
        <div class="message-label">You</div>
        <div class="message-content">${escapeHtml(question)}</div>
    `;
    chatContainer.appendChild(userMessageDiv);

    // Add assistant message placeholder
    const assistantMessageDiv = document.createElement('div');
    assistantMessageDiv.className = 'message assistant-message';
    assistantMessageDiv.innerHTML = `
        <div class="message-label">AI Assistant</div>
        <div class="message-content">
            <div class="loading-spinner" style="padding: 1rem;">
                <div class="spinner" style="width: 24px; height: 24px; border-width: 2px;"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(assistantMessageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    questionInput.value = '';

    try {
        // Get summary text
        const summaryContent = document.getElementById('summaryContent').innerText;

        // Stream answer
        const response = await fetch('/api/question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                video_title: currentVideoData.title,
                chunks_data: [], // We'll need to pass this from backend
                subtitle_text: '', // We'll need to pass this from backend
                summary: summaryContent,
                conversation_history: conversationHistory
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let answerText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(data);
                        answerText = parsed.accumulated;
                        assistantMessageDiv.querySelector('.message-content').innerHTML =
                            renderMarkdown(answerText);
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            }
        }

        // Update conversation history
        conversationHistory.push(`Q: ${question}`);
        conversationHistory.push(`A: ${answerText}`);
        if (conversationHistory.length > 8) {
            conversationHistory = conversationHistory.slice(-8);
        }

    } catch (error) {
        assistantMessageDiv.querySelector('.message-content').innerHTML =
            `<p style="color: #ef4444;">Error: ${error.message}</p>`;
    } finally {
        askBtn.disabled = false;
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

document.getElementById('askBtn').addEventListener('click', askQuestion);
document.getElementById('questionInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

// Utility function
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
initTheme();
