document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const startBtn = document.getElementById('start-btn');
    const videoFeed = document.getElementById('video-feed');
    const logTerminal = document.getElementById('log-terminal');

    const stopBtn = document.getElementById('stop-btn');
    const resetBtn = document.getElementById('reset-btn');

    const MAX_LOG_LINES = 200;
    const MAX_BUG_LINES = 100;   // the log terminal is capped; this was not,
                                 // so a long session could grow it unbounded
    let isTesting = false;
    let hasConnectedBefore = false;
    let currentFrameUrl = null;  // tracks the last object URL so it can be
                                  // revoked - otherwise each frame leaks
                                  // browser memory indefinitely

    // Blanks the video panel and releases the last frame's memory. Shared by
    // Reset and by a reconnect, which must both leave no stale frame behind.
    function clearFrame() {
        videoFeed.removeAttribute('src');
        if (currentFrameUrl) {
            URL.revokeObjectURL(currentFrameUrl);
            currentFrameUrl = null;
        }
    }

    startBtn.addEventListener('click', () => {
        isTesting = true;
        socket.emit('start_testing');
        startBtn.disabled = true;
        stopBtn.disabled = false;
        resetBtn.disabled = false;
        startBtn.textContent = 'Testing...';

        // Looked up by id rather than by searching innerHTML for the literal
        // text 'start testing...'. The old check broke silently if the
        // wording changed, and re-serialised the whole panel on every click.
        const bugPlaceholder = document.getElementById('bug-placeholder');
        if (bugPlaceholder) {
            bugPlaceholder.textContent = 'No bugs found yet...';
        }

        const logPlaceholder = document.getElementById('log-placeholder');
        if (logPlaceholder) {
            logPlaceholder.remove();
        }
    });

    // Puts the controls back to "not running". Shared by the Stop button and
    // by connection loss, so the two can never drift out of sync.
    function setIdleUI() {
        isTesting = false;
        stopBtn.disabled = true;
        startBtn.disabled = false;
        startBtn.textContent = 'START TESTING';
    }

    // Builds the italic grey "start testing..." line both panels show when
    // idle. Styling lives in .placeholder in style.css, not inline here.
    function makePlaceholder(tag, id) {
        const el = document.createElement(tag);
        el.id = id;
        el.className = 'placeholder';
        el.textContent = 'start testing...';
        return el;
    }

    function note(message) {
        const p = document.createElement('p');
        p.textContent = message;
        p.className = 'placeholder';
        logTerminal.appendChild(p);
        logTerminal.scrollTop = logTerminal.scrollHeight;
    }

    stopBtn.addEventListener('click', () => {
        socket.emit('stop_testing');
        setIdleUI();
    });

    resetBtn.addEventListener('click', () => {
        isTesting = false;
        socket.emit('stop_testing');
        socket.emit('reset_game');

        logTerminal.replaceChildren(makePlaceholder('p', 'log-placeholder'));
        const bugList = document.getElementById('bug-list');
        if (bugList) {
            bugList.replaceChildren(makePlaceholder('li', 'bug-placeholder'));
        }

        clearFrame();

        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        startBtn.textContent = 'START TESTING';
    });

    socket.on('video_frame', (data) => {
        if (!isTesting) return;
        // The server now sends the JPEG as raw binary (an ArrayBuffer),
        // not a base64 string - faster to produce server-side and smaller
        // over the wire. An ArrayBuffer is always truthy even when empty,
        // so check byteLength rather than the value itself.
        if (data.frame && data.frame.byteLength > 0) {
            const blob = new Blob([data.frame], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            videoFeed.src = url;
            // Revoke the PREVIOUS frame's URL now that a new one is set -
            // otherwise every frame permanently pins its memory, and at
            // 60fps that's a leak of tens of URLs per second.
            if (currentFrameUrl) {
                URL.revokeObjectURL(currentFrameUrl);
            }
            currentFrameUrl = url;
        }
    });

    socket.on('agent_log', (data) => {
        if (!isTesting) return;

        // Handle Bug Tracking UI
        if (data.log && data.log.includes('🚨 BUG FOUND:')) {
            const bugList = document.getElementById('bug-list');
            if (bugList) {
                // Drop the placeholder by id rather than by matching its
                // text - the old check compared against the literal string
                // 'No bugs', so rewording the placeholder would silently
                // leave it stuck above the first real entry.
                const placeholder = bugList.querySelector('#bug-placeholder');
                if (placeholder) {
                    placeholder.remove();
                }

                // The server sends "Step N: 🚨 BUG FOUND: <what happened>".
                const stepMatch = data.log.match(/^Step (\d+):/);
                const bugText = data.log.split('🚨 BUG FOUND: ')[1] || data.log;

                // Built with textContent, not innerHTML. These strings are
                // assembled server-side from live game state, so anything
                // that ever ends up looking like markup would otherwise be
                // parsed as HTML instead of shown as text.
                const li = document.createElement('li');
                li.className = 'bug-entry';
                const step = document.createElement('strong');
                step.textContent = `Step ${stepMatch ? stepMatch[1] : '?'}: `;
                li.appendChild(step);
                li.appendChild(document.createTextNode(bugText));
                bugList.appendChild(li);

                while (bugList.children.length > MAX_BUG_LINES) {
                    bugList.removeChild(bugList.firstChild);
                }

                bugList.parentElement.scrollTop = bugList.parentElement.scrollHeight;
            }
        }

        const p = document.createElement('p');
        p.textContent = data.log;
        logTerminal.appendChild(p);

        // Cap the log terminal
        while (logTerminal.children.length > MAX_LOG_LINES) {
            logTerminal.removeChild(logTerminal.firstChild);
        }

        // Auto-scroll
        logTerminal.scrollTop = logTerminal.scrollHeight;
    });

    // A pause the SERVER initiated - most often the user closing the game
    // window with its X. The session is kept, so START TESTING reopens the
    // window and carries on from the same moment.
    const PAUSE_NOTES = {
        window_closed: '— game window closed: testing paused. Press START TESTING to reopen it and resume —',
        session_ended: '— the agent session ended —',
        error: '— testing paused after an error (see the server console) —',
    };
    socket.on('testing_paused', (data) => {
        const reason = (data && data.reason) || '';
        note(PAUSE_NOTES[reason] || '— testing paused —');
        setIdleUI();
    });

    // ─── CONNECTION LIFECYCLE ───
    // Without these the dashboard could sit showing "Testing..." with the
    // Start button greyed out long after the stream had actually stopped.
    // The server pauses whenever a client connects or disconnects (keeping
    // the session and the game window), so after any reconnect it is
    // definitively NOT running - the UI has to agree, or the only way out is
    // a manual page refresh. START TESTING then resumes the same session.
    socket.on('disconnect', () => {
        if (isTesting) {
            note('— connection lost, testing stopped —');
        }
        setIdleUI();
        resetBtn.disabled = true;
    });

    socket.on('connect', () => {
        if (hasConnectedBefore) {
            note('— reconnected, press START TESTING to resume —');
            setIdleUI();
            resetBtn.disabled = true;
            clearFrame();
        }
        hasConnectedBefore = true;
    });

    // Info Modal Logic
    const infoBtn = document.getElementById('info-btn');
    const infoModal = document.getElementById('info-modal');
    const closeModal = document.querySelector('.close-modal');

    if (infoBtn && infoModal && closeModal) {
        // A closed modal is only faded out (.hidden animates opacity), so it
        // is also made inert: otherwise Tab still reaches its invisible ×.
        // Opening moves focus into it; closing hands focus back to the i.
        function setModalOpen(open) {
            const wasOpen = !infoModal.classList.contains('hidden');
            infoModal.classList.toggle('hidden', !open);
            infoModal.inert = !open;
            if (open) {
                closeModal.focus();
            } else if (wasOpen) {
                infoBtn.focus();
            }
        }

        infoBtn.addEventListener('click', () => setModalOpen(true));
        closeModal.addEventListener('click', () => setModalOpen(false));

        // The × is a focusable role="button" span, so it has to answer the
        // keys a real button does - otherwise a keyboard user can tab to it
        // and nothing happens.
        closeModal.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setModalOpen(false);
            }
        });

        // Close when clicking outside
        infoModal.addEventListener('click', (e) => {
            if (e.target === infoModal) {
                setModalOpen(false);
            }
        });

        // ...and on Escape, which is what people reflexively press.
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                setModalOpen(false);
            }
        });
    }
});
