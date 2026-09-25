document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const startBtn = document.getElementById('start-btn');
    const videoFeed = document.getElementById('video-feed');
    const logTerminal = document.getElementById('log-terminal');

    const stopBtn = document.getElementById('stop-btn');
    const resetBtn = document.getElementById('reset-btn');

    const MAX_LOG_LINES = 200;
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

    // ─── INCIDENTS (Objective 3) ───
    // Everything below renders what the SERVER says: /api/status for whether
    // testing stopped on a bug, /api/incidents for the history. Nothing here
    // decides that a bug was found - it only shows it. All text goes in via
    // textContent, never innerHTML: it is assembled from live game state.
    const bugBanner = document.getElementById('bug-banner');
    const bugBannerBody = document.getElementById('bug-banner-body');
    const bugList = document.getElementById('bug-list');
    const envSelect = document.getElementById('env-select');
    const syntheticBadge = document.getElementById('synthetic-badge');

    const FILE_LABELS = [
        ['report.pdf', 'PDF report'], ['report.md', 'Markdown report'],
        ['trigger.png', 'Trigger frame'], ['context.gif', 'GIF'],
    ];

    function el(tag, className, text) {
        const e = document.createElement(tag);
        if (className) e.className = className;
        if (text !== undefined) e.textContent = text;
        return e;
    }

    function fileUrl(id, name, download) {
        return `/incidents/${encodeURIComponent(id)}/${encodeURIComponent(name)}` +
               (download ? '?download=1' : '');
    }

    function localTime(utc) {
        const d = new Date(utc);
        return isNaN(d) ? utc : d.toLocaleString();
    }

    // Open / download links for whatever the bundle already holds. A report
    // still rendering is shown as such rather than as a dead link.
    function evidenceLinks(inc) {
        const box = el('div', 'evidence-links');
        const latest = inc.latest || {};
        for (const [base, label] of FILE_LABELS) {
            const name = latest[base] || (inc.available.includes(base) ? base : null);
            if (name) {
                const open = el('a', 'evidence-link', label);
                open.href = fileUrl(inc.incident_id, name, false);
                open.target = '_blank';
                open.rel = 'noopener';
                box.appendChild(open);
            } else {
                const status = (inc.renders || {})[base];
                box.appendChild(el('span', 'evidence-pending',
                    `${label}: ${status === 'failed' ? 'failed' : status === 'skipped' ? 'n/a' : 'rendering...'}`));
            }
        }
        const bundle = el('a', 'evidence-link evidence-link--bundle', 'Download all (.zip)');
        bundle.href = `/incidents/${encodeURIComponent(inc.incident_id)}/bundle.zip`;
        box.appendChild(bundle);
        return box;
    }

    function incidentFacts(inc) {
        const facts = el('dl', 'incident-facts');
        const rows = [
            ['Type', inc.category], ['Incident', inc.incident_id],
            ['Captured', localTime(inc.created_utc)],
            ['Where', `world x ${inc.location.x}, y ${inc.location.y}`],
            ['Severity', inc.severity], ['Confidence', inc.confidence],
            ['Reproduced', inc.reproduction], ['Seen', `${inc.occurrences}x`],
        ];
        for (const [k, v] of rows) {
            facts.appendChild(el('dt', null, k));
            facts.appendChild(el('dd', null, String(v)));
        }
        return facts;
    }

    function incidentCard(inc, compact) {
        const li = el(compact ? 'li' : 'div', 'bug-entry incident' + (inc.synthetic ? ' incident--synthetic' : ''));
        if (inc.synthetic) li.appendChild(el('p', 'synthetic-tag', 'SYNTHETIC TEST - not a game bug'));
        li.appendChild(el('strong', 'incident-title', inc.title));
        li.appendChild(el('p', 'incident-desc', inc.description));
        li.appendChild(incidentFacts(inc));
        li.appendChild(evidenceLinks(inc));
        return li;
    }

    let bannerIds = [];
    function showBugBanner(incidents) {
        bannerIds = incidents.map((i) => i.incident_id);
        bugBannerBody.replaceChildren(...incidents.map((i) => incidentCard(i, false)));
        bugBanner.classList.remove('hidden');
    }

    function hideBugBanner() {
        bannerIds = [];
        bugBanner.classList.add('hidden');
        bugBannerBody.replaceChildren();
    }

    // ─── CLEAN-GAME RUN END ───
    // When a run of the clean game ends the server stops testing and says how
    // (run_finished / /api/status run_result). A run that reached the castle
    // also has a run report - "No Bugs Found" when no detector fired - listed
    // in the Bug Tracker until Reset, like an incident.
    const runBanner = document.getElementById('run-banner');
    const runBannerTitle = document.getElementById('run-banner-title');
    const runBannerBody = document.getElementById('run-banner-body');
    const RUN_TITLES = {
        level_complete: 'Level Complete',
        death: 'Testing Stopped - Mario Died',
        timeout: 'Testing Stopped - Time Ran Out',
        safety_reset: 'Testing Stopped - The Agent Got Stuck',
    };
    const RUN_FILES = [
        ['report.pdf', 'PDF report'], ['report.md', 'Markdown report'],
        ['final.png', 'Final frame'], ['finish.gif', 'GIF'],
    ];
    let sessionRuns = [];      // this session's run reports, oldest first

    function runLinks(report) {
        const box = el('div', 'evidence-links');
        for (const [name, label] of RUN_FILES) {
            if ((report.available || []).includes(name)) {
                const a = el('a', 'evidence-link', label);
                a.href = `/runs/${encodeURIComponent(report.run_id)}/${encodeURIComponent(name)}`;
                a.target = '_blank';
                a.rel = 'noopener';
                box.appendChild(a);
            } else {
                const status = (report.renders || {})[name];
                box.appendChild(el('span', 'evidence-pending',
                    `${label}: ${status === 'failed' ? 'failed' : 'rendering...'}`));
            }
        }
        const bundle = el('a', 'evidence-link evidence-link--bundle', 'Download all (.zip)');
        bundle.href = `/runs/${encodeURIComponent(report.run_id)}/bundle.zip`;
        box.appendChild(bundle);
        return box;
    }

    function runCard(report) {
        const li = el('li', 'bug-entry run-entry');
        li.appendChild(el('strong', 'incident-title', `Run report - ${report.headline}`));
        li.appendChild(el('p', 'incident-desc',
            `${report.outcome} after ${report.agent_steps} agent steps (${report.run_id}).`));
        li.appendChild(runLinks(report));
        return li;
    }

    function rememberRun(report) {
        if (!report) return;
        const i = sessionRuns.findIndex((r) => r.run_id === report.run_id);
        if (i >= 0) sessionRuns[i] = report; else sessionRuns.push(report);
    }

    function showRunBanner(result) {
        const report = result.report;
        const pass = !!report && report.bugs === 0;
        runBannerTitle.textContent = (RUN_TITLES[result.end_reason] || 'Testing Stopped - Run Ended')
            + (report ? ` - ${report.headline}` : '');
        runBanner.classList.toggle('run-banner--pass', pass);
        runBannerBody.replaceChildren();
        if (report) {
            runBannerBody.appendChild(el('p', 'incident-desc', pass
                ? 'Mario reached the castle and none of the detectors fired. The run report is saved.'
                : 'Mario reached the castle. The run report is saved.'));
            runBannerBody.appendChild(runLinks(report));
        }
        runBanner.classList.remove('hidden');
    }

    function hideRunBanner() {
        runBanner.classList.add('hidden');
        runBannerBody.replaceChildren();
    }

    // With nothing recorded the tracker mirrors the log: "start testing..."
    // until testing has started, then "No bugs found yet".
    let testingStarted = false;
    let lastIncidents = [];
    function renderHistory(incidents) {
        lastIncidents = incidents;
        const runs = sessionRuns.map(runCard);
        if (!incidents.length) {
            bugList.replaceChildren(...(runs.length ? runs : [el('li', 'placeholder',
                testingStarted ? 'No bugs found yet...' : 'start testing...')]));
            return;
        }
        bugList.replaceChildren(...incidents.map((i) => incidentCard(i, true)), ...runs);
        // Keep the banner's links current as its reports finish rendering.
        if (bannerIds.length) {
            const current = incidents.filter((i) => bannerIds.includes(i.incident_id));
            if (current.length) showBugBanner(current);
        }
    }

    async function refreshIncidents() {
        try {
            const res = await fetch('/api/incidents', { cache: 'no-store' });
            if (res.ok) renderHistory((await res.json()).incidents || []);
        } catch (err) {
            console.warn('could not load incidents', err);
        }
    }

    async function refreshStatus() {
        try {
            const res = await fetch('/api/status', { cache: 'no-store' });
            if (!res.ok) return;
            const s = await res.json();
            if (s.game_variant) envSelect.value = s.game_variant;
            envSelect.disabled = false;
            if (s.testing || s.steps > 0) setTestingStarted(true);
            syntheticBadge.classList.toggle('hidden', !(s.synthetic_probes || []).length);
            if (s.bug_found && s.bug_found.length) {
                showBugBanner(s.bug_found);
            } else {
                hideBugBanner();
            }
            if (s.run_result) {
                rememberRun(s.run_result.report);
                showRunBanner(s.run_result);
                renderHistory(lastIncidents);
            } else {
                hideRunBanner();
            }
            if (!s.testing) setIdleUI();
        } catch (err) {
            console.warn('could not load status', err);
        }
    }

    startBtn.addEventListener('click', () => {
        isTesting = true;
        socket.emit('start_testing');
        startBtn.disabled = true;
        stopBtn.disabled = false;
        resetBtn.disabled = false;
        startBtn.textContent = 'Testing...';
        hideBugBanner();     // the server clears bug_found on resume, and says so
        hideRunBanner();     // ...and run_result: Start plays the next run
        setTestingStarted(true);

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

    function resetDashboard() {
        isTesting = false;
        socket.emit('stop_testing');
        socket.emit('reset_game');

        // Reset ends the session: the log and the Bug Tracker start empty
        // (the server starts a new session list; every incident stays saved).
        logTerminal.replaceChildren(makePlaceholder('p', 'log-placeholder'));
        hideBugBanner();
        hideRunBanner();
        lastIncidents = [];
        sessionRuns = [];
        setTestingStarted(false);

        clearFrame();

        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        startBtn.textContent = 'START TESTING';
    }
    resetBtn.addEventListener('click', resetDashboard);

    function setTestingStarted(started) {
        testingStarted = started;
        renderHistory(lastIncidents);
    }

    // Picking the other game resets the dashboard, exactly like Reset, and
    // the server loads that game; START TESTING then plays it.
    envSelect.addEventListener('change', () => {
        const variant = envSelect.value;
        resetDashboard();
        envSelect.disabled = true;
        startBtn.disabled = true;
        startBtn.textContent = 'Loading game...';
        socket.emit('switch_game', { variant });
    });

    socket.on('game_switched', (data) => {
        if (!(data && data.ok)) {
            note(`— could not switch the game: ${(data && data.error) || 'unknown error'} —`);
        }
        refreshStatus();       // the server's selection wins, whichever it is
        startBtn.disabled = false;
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

        // The detector's own line ("Step N: 🚨 BUG FOUND: ...") stays in the
        // log. The BUG TRACKER no longer parses it: it lists the incidents the
        // server actually recorded (see renderHistory), with their evidence.
        const p = document.createElement('p');
        p.textContent = data.log;
        if (data.log && data.log.includes('🚨 BUG FOUND:')) {
            p.className = 'log-bug';
        }
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
        bug_found: '— BUG FOUND: testing stopped. The evidence is saved; press START TESTING to continue —',
        capture_failed: '— an anomaly was detected but its evidence could not be saved (see the server console) —',
        level_complete: '— LEVEL COMPLETE: testing stopped. Press RESET DASHBOARD, then START TESTING, for a new run —',
        mario_died: '— MARIO DIED: testing stopped. Press RESET DASHBOARD, then START TESTING, for a new run —',
        run_ended: '— the run ended: testing stopped. Press RESET DASHBOARD, then START TESTING, for a new run —',
    };
    socket.on('testing_paused', (data) => {
        const reason = (data && data.reason) || '';
        note(PAUSE_NOTES[reason] || '— testing paused —');
        setIdleUI();
    });

    socket.on('bug_found', (data) => {
        showBugBanner((data && data.incidents) || []);
        setIdleUI();
        refreshIncidents();
    });
    socket.on('bug_cleared', hideBugBanner);
    socket.on('run_finished', (result) => {
        if (!result) return;
        rememberRun(result.report);
        showRunBanner(result);
        setIdleUI();
        renderHistory(lastIncidents);
    });
    socket.on('run_cleared', hideRunBanner);
    socket.on('run_report_updated', (report) => {     // its GIF/MD/PDF finished rendering
        if (!report || !sessionRuns.some((r) => r.run_id === report.run_id)) return;
        rememberRun(report);
        renderHistory(lastIncidents);
        if (!runBanner.classList.contains('hidden')) {
            runBannerBody.querySelectorAll('.evidence-links').forEach((box) =>
                box.replaceWith(runLinks(report)));
        }
    });
    socket.on('incident_updated', refreshIncidents);      // a report finished rendering
    socket.on('incident_occurrence', (inc) => {
        note(`— seen again: ${inc.title} (${inc.incident_id}), now ${inc.occurrences}x —`);
        refreshIncidents();
    });
    socket.on('incident_capture_failed', (data) => {
        note(`— anomaly detected, but its evidence could not be saved: ${(data && data.error) || 'unknown error'} —`);
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
            // A dropped connection came back: nothing was reset, so the same
            // session resumes on START TESTING.
            note('— reconnected, press START TESTING to resume —');
            setIdleUI();
            resetBtn.disabled = true;
            clearFrame();
            refreshStatus();
            refreshIncidents();
            return;
        }
        hasConnectedBefore = true;
        // A freshly loaded page (a refresh, a reopened tab) starts from a clean
        // dashboard, exactly like RESET DASHBOARD. The server answers once the
        // reset has run; only then is its state read, so no old banner, report
        // or bug is shown again.
        socket.emit('page_opened', () => {
            refreshStatus();
            refreshIncidents();
        });
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
