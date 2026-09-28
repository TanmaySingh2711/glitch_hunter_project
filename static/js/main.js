'use strict';
// The Glitch Hunter dashboard page.
//
// Everything it shows is the SERVER's truth: /api/status (running, paused,
// the bug that stopped testing, how a clean run ended, the live panel),
// /api/project (the project's facts, read from its own files), /api/incidents
// and /api/runs (the evidence on disk), and the Socket.IO events. Nothing
// here decides that a bug was found or invents a number - it only shows what
// it was told. All text goes in via textContent, never innerHTML: much of it
// is assembled from live game state.
document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const $ = (id) => document.getElementById(id);

    const startBtn = $('start-btn');
    const stopBtn = $('stop-btn');
    const resetBtn = $('reset-btn');
    const videoFeed = $('video-feed');
    const logTerminal = $('log-terminal');
    const bugList = $('bug-list');
    const gameRadios = [...document.querySelectorAll('input[name="game"]')];

    const MAX_LOG_LINES = 200;
    const GAME_NAMES = { mario_clean: 'Clean game', mario_bugged: 'Bugged game' };

    // ─── STATE (mirrors the server; see refreshStatus and the socket events) ───
    const state = {
        connected: false,
        everConnected: false,
        testing: false,          // frames and log lines are shown only while true
        steps: 0,                // agent steps in this session (0 = fresh session)
        pauseReason: null,
        switching: false,
        bugFound: null,          // the incident(s) that stopped testing
        runResult: null,         // how the last clean-game run ended
        status: {},              // the last /api/status (game, brain, ...)
        telemetry: null,         // the live panel's last values
        sessionIncidents: [],    // this session's incidents (Bug Tracker)
        sessionRuns: [],         // this session's run reports, oldest first
        facts: null,             // /api/project
        evidenceChoice: 'auto',  // what the monitor shows while stopped
    };
    let currentFrameUrl = null;  // the last object URL, revoked when replaced -
                                 // otherwise each frame leaks browser memory

    // ─── SMALL HELPERS ───
    function el(tag, className, text) {
        const e = document.createElement(tag);
        if (className) e.className = className;
        if (text !== undefined && text !== null) e.textContent = text;
        return e;
    }

    const ICONS = {
        check: ['M5 12.5l4.5 4.5L19 7.5'],
        alert: ['M12 3.5 2.8 19.5h18.4Z', 'M12 10v4.5', 'M12 17.3v.2'],
        bug: ['M8.5 9.5h7v5a3.5 3.5 0 0 1-7 0Z', 'M12 9.5v8.5', 'M5 13.5h3.5M15.5 13.5H19',
              'M6 8.5l2.5 1.5M18 8.5l-2.5 1.5', 'M9.5 7a2.5 2.5 0 0 1 5 0v2.5h-5Z'],
        pause: ['M9 6v12', 'M15 6v12'],
        play: ['M8 5.5v13l10.5-6.5Z'],
        flag: ['M6 21V4', 'M6 4.5h11l-2.5 4 2.5 4H6'],
        clock: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z', 'M12 7.5V12l3 2'],
        dot: ['M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z'],
        ring: ['M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Z'],
        offline: ['M3 3l18 18', 'M8.5 16.5a5 5 0 0 1 7 0', 'M5 12.5a10 10 0 0 1 4-2.3',
                  'M19 12.5a10 10 0 0 0-5.3-2.7', 'M12 20h.01'],
        brain: ['M12 4v16', 'M12 6.5A3 3 0 0 0 6.5 7 3 3 0 0 0 5 12a3 3 0 0 0 2 4.5A3 3 0 0 0 12 18',
                'M12 6.5A3 3 0 0 1 17.5 7 3 3 0 0 1 19 12a3 3 0 0 1-2 4.5A3 3 0 0 1 12 18'],
        search: ['M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13Z', 'M15.5 15.5 20 20'],
        file: ['M7 3h7l4 4v14H7Z', 'M14 3v4h4', 'M9.5 12h6M9.5 15.5h6'],
        replay: ['M4.5 12a7.5 7.5 0 1 0 2.2-5.3', 'M4 4v4h4'],
    };
    const FILLED = new Set(['play', 'dot']);

    function icon(name) {
        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('aria-hidden', 'true');
        svg.setAttribute('fill', FILLED.has(name) ? 'currentColor' : 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '2');
        svg.setAttribute('stroke-linecap', 'round');
        svg.setAttribute('stroke-linejoin', 'round');
        for (const d of ICONS[name] || ICONS.dot) {
            const p = document.createElementNS(ns, 'path');
            p.setAttribute('d', d);
            svg.appendChild(p);
        }
        return svg;
    }

    const num = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : '—');
    const pct = (x, digits = 1) => (typeof x === 'number' ? `${(x * 100).toFixed(digits)}%` : '—');

    function localTime(utc) {
        const d = new Date(utc);
        return isNaN(d) ? String(utc) : d.toLocaleString();
    }

    function shortTime(utc) {
        const d = new Date(utc);
        return isNaN(d) ? String(utc) : d.toLocaleTimeString();
    }

    function link(label, href, opts = {}) {
        const a = el('a', `btn ${opts.primary ? 'btn--primary' : 'btn--secondary'} btn--sm`, label);
        a.href = href;
        if (!opts.download) {
            a.target = '_blank';
            a.rel = 'noopener';
        }
        return a;
    }

    function pendingChip(text, failed) {
        const s = el('span', 'pending-chip');
        if (!failed) s.appendChild(el('span', 'spinner'));
        s.appendChild(document.createTextNode(text));
        return s;
    }

    function checkItem(kind, title, sub) {
        // kind: ok | wait | warn | bug | idle
        const li = el('li');
        const ic = el('span', `check-icon check-icon--${kind}`);
        if (kind === 'wait') {
            ic.appendChild(el('span', 'spinner'));
        } else {
            ic.appendChild(icon({ ok: 'check', warn: 'alert', bug: 'bug', idle: 'ring' }[kind] || 'dot'));
        }
        li.appendChild(ic);
        const t = el('span', 'checklist__text');
        t.appendChild(el('b', null, title));
        if (sub) t.appendChild(el('span', 'checklist__sub', sub));
        li.appendChild(t);
        return li;
    }

    function setFact(name, text) {
        document.querySelectorAll(`[data-fact="${name}"]`).forEach((n) => { n.textContent = text; });
    }

    function detectorCount() {
        const f = state.facts;
        return f && f.objective3 ? f.objective3.detectors.length : null;
    }

    // ─── INCIDENT AND RUN FILES ───
    function fileUrl(id, name, download) {
        return `/incidents/${encodeURIComponent(id)}/${encodeURIComponent(name)}` +
               (download ? '?download=1' : '');
    }

    function runFileUrl(id, name) {
        return `/runs/${encodeURIComponent(id)}/${encodeURIComponent(name)}`;
    }

    // The newest rendered version of one of an incident's files, or null.
    function incidentFile(inc, base) {
        return (inc.latest || {})[base] || ((inc.available || []).includes(base) ? base : null);
    }

    function renderState(inc, base) {
        return (inc.renders || {})[base] || 'pending';
    }

    const REPLAY_TEXT = {
        reproduced: ['Reproduced', 'Replayed in a separate process: the same state, the same detector on the same frame, identical pixels.'],
        reproduced_state_only: ['Reproduced (state)', 'Replayed: the same state and the same detector on the same frame; the pixels differ slightly.'],
        not_reproduced: ['Not reproduced', 'The replay matched the run, but the detector stayed quiet.'],
        diverged: ['Diverged', 'The replay separated from the recording before the trigger.'],
        not_possible: ['Not possible', 'There was nothing honest to replay for this incident.'],
        timeout: ['Timed out', 'The replay process did not finish in time.'],
        error: ['Replay failed', 'The replay process failed.'],
        not_attempted: ['Checking…', 'The run is being replayed in a separate process to confirm the bug.'],
    };

    function replayInfo(inc) {
        // 'pending' / 'not_attempted': the replay has not reported back yet.
        const raw = inc.reproduction || 'not_attempted';
        const status = raw === 'pending' || raw === 'running' ? 'not_attempted' : raw;
        if (status === 'not_attempted' && renderState(inc, 'reproduction.json') === 'skipped') {
            return { status: 'skipped', label: 'Skipped', text: 'Replay is switched off for this dashboard (--no-reproduce).', kind: 'idle' };
        }
        const [label, text] = REPLAY_TEXT[status] || [status, ''];
        const kind = status === 'not_attempted' ? 'wait'
            : status.startsWith('reproduced') ? 'ok' : 'warn';
        return { status, label, text, kind };
    }

    function reportsInfo(inc) {
        const names = ['context.gif', 'report.md', 'report.pdf'];
        const states = names.map((n) => (incidentFile(inc, n) ? 'done' : renderState(inc, n)));
        if (states.every((s) => s === 'done')) return { kind: 'ok', text: 'GIF, Markdown and PDF are ready' };
        if (states.some((s) => s === 'failed')) return { kind: 'warn', text: 'Some reports could not be written' };
        return { kind: 'wait', text: 'Writing the GIF, Markdown and PDF…' };
    }

    function incidentLinks(inc, withDetails) {
        const box = el('div', 'actions');
        const pdf = incidentFile(inc, 'report.pdf');
        if (pdf) box.appendChild(link('Open PDF report', fileUrl(inc.incident_id, pdf), { primary: true }));
        else box.appendChild(pendingChip('PDF report: ' + (renderState(inc, 'report.pdf') === 'failed' ? 'failed' : 'writing…'),
                                         renderState(inc, 'report.pdf') === 'failed'));
        const md = incidentFile(inc, 'report.md');
        if (md) box.appendChild(link('Markdown', fileUrl(inc.incident_id, md)));
        const gif = incidentFile(inc, 'context.gif');
        if (gif) box.appendChild(link('GIF', fileUrl(inc.incident_id, gif)));
        if (incidentFile(inc, 'trigger.png')) box.appendChild(link('Trigger frame', fileUrl(inc.incident_id, 'trigger.png')));
        box.appendChild(link('Download all (.zip)', `/incidents/${encodeURIComponent(inc.incident_id)}/bundle.zip`, { download: true }));
        if (withDetails) {
            const b = el('button', 'btn btn--secondary btn--sm', 'Full details');
            b.type = 'button';
            b.addEventListener('click', () => openIncident(inc.incident_id));
            box.appendChild(b);
        }
        return box;
    }

    const RUN_FILES = [['report.pdf', 'Open PDF report'], ['report.md', 'Markdown'],
                       ['final.png', 'Final frame'], ['finish.gif', 'GIF']];

    function runLinks(report) {
        const box = el('div', 'actions');
        for (const [name, label] of RUN_FILES) {
            if ((report.available || []).includes(name)) {
                box.appendChild(link(label, runFileUrl(report.run_id, name), { primary: name === 'report.pdf' }));
            } else {
                const failed = (report.renders || {})[name] === 'failed';
                box.appendChild(pendingChip(`${label.replace('Open ', '')}: ${failed ? 'failed' : 'writing…'}`, failed));
            }
        }
        box.appendChild(link('Download all (.zip)', `/runs/${encodeURIComponent(report.run_id)}/bundle.zip`, { download: true }));
        return box;
    }

    function sevChip(sev) {
        return el('span', `sev sev--${sev || 'none'}`, sev || 'n/a');
    }

    // ─── NAVIGATION (tabs; a page load would reset the dashboard) ───
    const tabs = [...document.querySelectorAll('.tab')];
    const VIEWS = tabs.map((t) => t.dataset.view);

    function showView(view, focusTab) {
        if (!VIEWS.includes(view)) view = 'overview';
        for (const t of tabs) {
            const on = t.dataset.view === view;
            t.setAttribute('aria-selected', String(on));
            t.tabIndex = on ? 0 : -1;
            $(t.getAttribute('aria-controls')).hidden = !on;
            if (on && focusTab) t.focus();
        }
        if (location.hash !== `#${view}`) history.replaceState(null, '', `#${view}`);
        if (view === 'history') refreshHistory();
        if (view === 'overview' || view === 'method') refreshFacts();
    }

    tabs.forEach((t, i) => {
        t.addEventListener('click', () => showView(t.dataset.view));
        t.addEventListener('keydown', (e) => {
            const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
            if (d) {
                e.preventDefault();
                showView(tabs[(i + d + tabs.length) % tabs.length].dataset.view, true);
            }
        });
    });
    document.querySelectorAll('[data-goto]').forEach((b) => b.addEventListener('click', (e) => {
        e.preventDefault();
        showView(b.dataset.goto);
    }));

    // ─── WHAT STATE IS THE SYSTEM IN? ───
    function phase() {
        if (!state.connected) return state.everConnected ? 'offline' : 'connecting';
        if (state.switching) return 'switching';
        if (state.bugFound && state.bugFound.length) return 'bug';
        if (state.runResult) {
            const r = state.runResult;
            return r.end_reason === 'level_complete' && r.report && r.report.bugs === 0 ? 'clean' : 'ended';
        }
        if (state.testing) return 'testing';
        if (state.pauseReason === 'error') return 'error';
        if (state.pauseReason === 'capture_failed') return 'capture_failed';
        if (state.steps > 0) return 'paused';
        return 'ready';
    }

    const PILLS = {
        connecting: ['neutral', 'ring', 'Connecting…'],
        offline: ['warn', 'offline', 'Disconnected'],
        switching: ['info', 'clock', 'Loading game'],
        bug: ['bug', 'bug', 'Bug found'],
        clean: ['ok', 'check', 'No bugs found'],
        ended: ['neutral', 'flag', 'Run ended'],
        testing: ['info', null, 'Testing'],
        error: ['warn', 'alert', 'Stopped: error'],
        capture_failed: ['warn', 'alert', 'Evidence not saved'],
        paused: ['neutral', 'pause', 'Paused'],
        ready: ['neutral', 'dot', 'Ready'],
    };

    function renderPill(p) {
        const [kind, ic, text] = PILLS[p];
        const pill = $('system-pill');
        pill.className = `pill pill--${kind}`;
        const iconBox = pill.querySelector('.pill__icon');
        iconBox.replaceChildren(ic ? icon(ic) : el('span', 'live-dot'));
        pill.querySelector('.pill__text').textContent = text;

        const conn = $('conn-indicator');
        conn.className = `conn ${state.connected ? 'conn--ok' : state.everConnected ? 'conn--lost' : 'conn--wait'}`;
        conn.querySelector('.conn__text').textContent = state.connected ? 'Connected'
            : state.everConnected ? 'Reconnecting…' : 'Connecting';

        const dot = $('tab-live-dot');
        dot.hidden = !['testing', 'bug', 'clean'].includes(p);
        dot.className = `tab__dot${p === 'bug' ? ' tab__dot--bug' : p === 'clean' ? ' tab__dot--ok' : ''}`;
    }

    function gameName() {
        return GAME_NAMES[state.status.game_variant] || 'game';
    }

    function renderControls(p) {
        const labels = {
            switching: 'Loading game…', testing: 'Testing…', bug: 'Resume testing',
            clean: 'Start next run', ended: 'Start next run', paused: 'Resume testing',
            error: 'Resume testing', capture_failed: 'Resume testing',
        };
        startBtn.textContent = labels[p] || 'Start testing';
        startBtn.disabled = !state.connected || state.switching || state.testing;
        stopBtn.disabled = !state.connected || !state.testing;
        resetBtn.disabled = !state.connected || state.switching
            || !(state.steps > 0 || state.testing || state.bugFound || state.runResult);
        for (const r of gameRadios) {
            r.checked = r.value === state.status.game_variant;
            r.disabled = !state.connected || state.switching;
        }
        const hints = {
            connecting: 'Connecting to the dashboard server…',
            offline: 'The dashboard server is not answering. The page reconnects by itself.',
            switching: 'Loading the other game and the AI brain against it…',
            ready: `Press Start testing: the AI starts playing the ${gameName().toLowerCase()} and its view appears in the centre.`,
            testing: 'Pause stops at once and keeps everything. Reset ends this session and clears the page.',
            paused: 'Resume continues from the same moment. Reset starts a fresh session.',
            bug: 'Resume testing continues from the moment of the bug. The evidence stays saved.',
            clean: 'Start next run plays the level again. Reset clears the page first.',
            ended: 'Start next run plays the level again. Reset clears the page first.',
            error: 'Testing stopped after an error (see the dashboard\'s terminal window).',
            capture_failed: 'A rule broke but its evidence could not be saved (see the terminal window).',
        };
        $('control-hint').textContent = hints[p] || '';
    }

    function renderNow(p) {
        const card = $('now-card');
        const d = detectorCount();
        const checks = d ? `${d} detectors` : 'The detectors';
        const t = state.telemetry || {};
        let kind = 'neutral';
        let title = '';
        let body = '';
        let next = '';
        switch (p) {
        case 'connecting':
            title = 'Connecting to the dashboard…';
            break;
        case 'offline':
            kind = 'warn';
            title = 'Connection lost';
            body = 'The page cannot reach the dashboard server. Check its terminal window; the page reconnects by itself.';
            break;
        case 'switching':
            kind = 'info';
            title = 'Loading the game…';
            body = 'The previous game is unloaded completely, then the AI brain is loaded against the new one.';
            break;
        case 'bug': {
            kind = 'bug';
            const inc = state.bugFound[0];
            title = `Bug found: ${inc.title}`;
            const replay = replayInfo(inc);
            const reports = reportsInfo(inc);
            body = 'Testing stopped automatically at the exact frame and the evidence is saved. '
                + (replay.kind === 'wait' ? 'The run is being replayed to confirm it'
                    : replay.kind === 'ok' ? 'The replay confirmed it' : `Replay: ${replay.label.toLowerCase()}`)
                + (reports.kind === 'ok' ? '; the GIF, Markdown and PDF reports are ready.'
                    : reports.kind === 'wait' ? ', and the reports are being written in the background.'
                        : '; some reports could not be written.');
            next = 'Look at the evidence, open the report, then press Resume testing to continue from this moment.';
            break;
        }
        case 'clean':
            kind = 'ok';
            title = 'Level complete. No bugs found.';
            body = `The AI reached the castle on the clean game and none of the ${d || ''} detectors fired. A run report has been saved.`.replace('  ', ' ');
            next = 'Open the run report, or press Start next run to play the level again.';
            break;
        case 'ended': {
            const r = state.runResult;
            title = RUN_TITLES[r.end_reason] || 'The run ended';
            body = r.end_reason === 'level_complete'
                ? 'The run reached the castle. See its report for what the detectors found.'
                : 'No detector fired before the run ended. On its own, a death is normal play, not a bug. Run reports are written only for runs that reach the castle.';
            next = 'Press Start next run to play the level again, or Reset to clear the page.';
            break;
        }
        case 'testing':
            kind = 'info';
            if (t.phase === 'explore') {
                title = 'The AI is exploring the level';
                body = `Its QA training pays it for reaching places it has never visited. ${checks} check the game's physics on every frame.`;
            } else if (t.phase === 'complete') {
                title = 'The AI is heading for the flag';
                body = `Exploring here has dried up, so it now moves towards the finish. ${checks} keep checking every frame.`;
            } else {
                title = `The AI is playing the ${gameName().toLowerCase()}`;
                body = `${checks} check the game's physics on every frame.`;
            }
            next = 'If any rule breaks, testing pauses by itself and a bug report is written.';
            break;
        case 'error':
            kind = 'warn';
            title = 'Testing stopped after an error';
            body = 'See the dashboard\'s terminal window for the details.';
            next = 'Press Resume testing to try again, or Reset to start fresh.';
            break;
        case 'capture_failed':
            kind = 'warn';
            title = 'A rule broke, but its evidence could not be saved';
            body = 'Testing stopped so that this is not missed. See the dashboard\'s terminal window.';
            break;
        case 'paused':
            title = state.pauseReason === 'window_closed' ? 'Paused: the game window was closed' : 'Testing paused';
            body = 'Nothing is lost: the AI, the run and everything found so far are kept.';
            next = 'Press Resume testing to continue from the same moment.';
            break;
        default:
            title = 'Ready to test';
            body = state.status.brain_path
                ? 'The AI brain is loaded. Choose a game on the left and press Start testing.'
                : 'Choose a game on the left and press Start testing.';
        }
        card.className = `now now--${kind}`;
        $('now-title').textContent = title;
        $('now-body').textContent = body;
        $('now-next').textContent = next;
    }

    // The five stages of a test, Objective 1 to 3, lit from real state only.
    function renderStepper(p) {
        const steps = [];
        const s = state.status;
        const brainOk = !!s.brain_path;
        steps.push({
            title: 'AI brain ready',
            sub: s.brain_approved ? 'Objectives 1 + 2: final QA brain' : brainOk ? 'Fallback brain' : 'Loading…',
            st: !state.connected && !state.everConnected ? 'pending' : s.brain_approved ? 'done' : brainOk ? 'warn' : 'pending',
        });
        const d = detectorCount();
        const ran = state.steps > 0;
        steps.push({
            title: p === 'paused' ? 'Exploring (paused)' : 'Exploring & checking',
            sub: `Objective 2 agent${d ? ` · ${d} checks per frame` : ''}`,
            st: p === 'testing' ? 'active' : ['bug', 'clean', 'ended'].includes(p) || ran ? 'done' : 'pending',
        });
        if (p === 'bug') {
            const inc = state.bugFound[0];
            const replay = replayInfo(inc);
            const reports = reportsInfo(inc);
            steps.push({ title: 'Bug detected', sub: `Objective 3 · ${inc.incident_id}`, st: 'bug' });
            steps.push({ title: 'Evidence & replay', sub: `Saved · replay: ${replay.label.toLowerCase()}`,
                         st: replay.kind === 'wait' ? 'active' : replay.kind === 'ok' ? 'done' : 'warn' });
            steps.push({ title: 'Report ready', sub: reports.kind === 'ok' ? 'GIF · Markdown · PDF' : reports.text,
                         st: reports.kind === 'ok' ? 'ok' : reports.kind === 'wait' ? 'active' : 'warn' });
        } else if (p === 'clean' || p === 'ended') {
            const r = state.runResult;
            const report = r.report;
            steps.push({ title: p === 'clean' ? 'Level complete' : 'Run ended',
                         sub: p === 'clean' ? 'Objective 3 · no detector fired' : (RUN_TITLES[r.end_reason] || ''),
                         st: p === 'clean' ? 'ok' : 'done' });
            const gifReady = report && (report.available || []).includes('finish.gif');
            steps.push({ title: 'Evidence saved', sub: report ? (gifReady ? 'Final frame · GIF' : 'Final frame · GIF writing…') : 'Only for runs that finish',
                         st: report ? (gifReady ? 'done' : 'active') : 'pending' });
            const pdf = report && (report.available || []).includes('report.pdf');
            steps.push({ title: 'Report ready', sub: report ? (pdf ? 'No Bugs Found report' : 'Writing…') : 'No report for this run',
                         st: report ? (pdf ? 'ok' : 'active') : 'pending' });
        } else {
            steps.push({ title: 'Bug detected?', sub: 'Objective 3 · waiting', st: 'pending' });
            steps.push({ title: 'Evidence & replay', sub: 'Saved at the exact frame', st: 'pending' });
            steps.push({ title: 'Report ready', sub: 'GIF · Markdown · PDF', st: 'pending' });
        }
        const ICON = { done: 'check', ok: 'check', bug: 'bug', warn: 'alert', pending: 'ring' };
        const STATUS_WORD = { done: 'done', ok: 'done', bug: 'bug found', warn: 'needs attention', active: 'in progress', pending: 'not yet' };
        $('stepper').replaceChildren(...steps.map((st, i) => {
            const li = el('li', `step step--${st.st}`);
            if (st.st === 'active') li.setAttribute('aria-current', 'step');
            const ic = el('span', 'step__icon');
            if (st.st === 'active') ic.appendChild(el('span', 'spinner'));
            else ic.appendChild(icon(ICON[st.st]));
            li.appendChild(ic);
            const tx = el('span', 'step__text');
            tx.appendChild(el('span', 'step__title', `${i + 1}. ${st.title}`));
            tx.appendChild(el('span', 'step__sub', st.sub));
            tx.appendChild(el('span', 'sr-only', ` (${STATUS_WORD[st.st]})`));
            li.appendChild(tx);
            return li;
        }));
    }

    const RUN_TITLES = {
        level_complete: 'Level complete',
        death: 'Mario died',
        timeout: 'Time ran out',
        safety_reset: 'The agent got stuck',
    };

    // ─── THE MONITOR: live frames, or the evidence while stopped ───
    function evidenceSources() {
        if (state.bugFound && state.bugFound.length) {
            const inc = state.bugFound[0];
            const gif = incidentFile(inc, 'context.gif');
            return [
                { key: 'live', label: 'Last live frame' },
                { key: 'trigger', label: 'Trigger frame', url: incidentFile(inc, 'trigger.png') ? fileUrl(inc.incident_id, 'trigger.png') : null,
                  alt: 'The exact frame the bug was detected on' },
                { key: 'gif', label: 'GIF: moments before', url: gif ? fileUrl(inc.incident_id, gif) : null,
                  pending: !gif, alt: 'The moments leading up to the bug' },
            ];
        }
        const report = state.runResult && state.runResult.report;
        if (report) {
            const has = (n) => (report.available || []).includes(n);
            return [
                { key: 'live', label: 'Last live frame' },
                { key: 'final', label: 'Final frame', url: has('final.png') ? runFileUrl(report.run_id, 'final.png') : null,
                  alt: 'The last frame of the run' },
                { key: 'gif', label: 'GIF: the finish', url: has('finish.gif') ? runFileUrl(report.run_id, 'finish.gif') : null,
                  pending: !has('finish.gif'), alt: 'The last seconds of the run' },
            ];
        }
        return [];
    }

    function renderMonitor(p) {
        const monitor = $('monitor');
        monitor.className = `monitor${p === 'testing' ? ' monitor--live' : p === 'bug' ? ' monitor--bug' : p === 'clean' ? ' monitor--ok' : ''}`;
        // Stopped with a result: the result and the session's bugs come first.
        document.querySelector('.live__side').classList.toggle('live__side--stopped', ['bug', 'clean', 'ended'].includes(p));
        const badge = $('monitor-badge');
        const B = {
            testing: ['live', null, 'LIVE · AI playing'],
            bug: ['bug', 'bug', 'BUG FOUND · stopped at the detection frame'],
            clean: ['ok', 'check', 'LEVEL COMPLETE · no bugs found'],
            paused: ['', 'pause', 'PAUSED'],
            ended: ['', 'flag', 'RUN ENDED'],
        }[p];
        badge.hidden = !B || !videoFeed.hasAttribute('src');
        if (B) {
            badge.className = `monitor__badge${B[0] ? ` monitor__badge--${B[0]}` : ''}`;
            badge.replaceChildren(B[1] ? icon(B[1]) : el('span', 'live-dot'), document.createTextNode(B[2]));
        }
        const game = $('monitor-game');
        game.hidden = !state.status.game_variant;
        game.textContent = gameName();
        $('monitor-empty').hidden = videoFeed.hasAttribute('src');

        // Stopped on a bug or a finished run: offer the saved evidence on the
        // monitor itself (the GIF by default, once it has been written).
        const sources = evidenceSources();
        const sw = $('evidence-switch');
        const view = $('evidence-view');
        if (!sources.length) {
            sw.hidden = true;
            view.hidden = true;
            view.removeAttribute('src');
            return;
        }
        let choice = state.evidenceChoice;
        if (choice === 'auto') {
            const gif = sources.find((s) => s.key === 'gif' && s.url);
            const still = sources.find((s) => s.key !== 'live' && s.key !== 'gif' && s.url);
            choice = (gif || still || sources[0]).key;
        }
        const chosen = sources.find((s) => s.key === choice && (s.key === 'live' || s.url)) || sources[0];
        sw.hidden = false;
        sw.replaceChildren(el('span', 'evidence-switch__label', 'Show:'), ...sources.map((s) => {
            const b = el('button', 'btn btn--secondary btn--sm', s.label + (s.pending ? ' (writing…)' : ''));
            b.type = 'button';
            b.disabled = s.key !== 'live' && !s.url;
            b.setAttribute('aria-pressed', String(s.key === chosen.key));
            b.addEventListener('click', () => { state.evidenceChoice = s.key; renderMonitor(phase()); });
            return b;
        }));
        monitor.classList.toggle('monitor--evidence', chosen.key !== 'live');
        if (chosen.key === 'live') {
            view.hidden = true;
        } else {
            if (view.getAttribute('src') !== chosen.url) view.src = chosen.url;
            view.alt = chosen.alt || '';
            view.hidden = false;
        }
    }

    function clearFrame() {
        videoFeed.removeAttribute('src');
        if (currentFrameUrl) {
            URL.revokeObjectURL(currentFrameUrl);
            currentFrameUrl = null;
        }
    }

    // ─── RESULT CARD: Bug Found, or how a clean run ended ───
    function renderResult(p) {
        const card = $('result-card');
        if (p === 'bug') {
            card.className = 'card result result--bug';
            card.replaceChildren(...state.bugFound.flatMap((inc, i) => {
                const parts = [];
                if (i === 0) {
                    const head = el('div', 'result__head');
                    const ic = el('span', 'result__icon');
                    ic.appendChild(icon('bug'));
                    head.appendChild(ic);
                    const tx = el('div');
                    tx.appendChild(el('p', 'result__kicker', 'Bug found · testing stopped automatically'));
                    tx.appendChild(el('p', 'result__title', state.bugFound.length > 1
                        ? `${state.bugFound.length} bugs on the same step` : inc.title));
                    head.appendChild(tx);
                    parts.push(head);
                }
                const body = el('div', 'result__body');
                if (state.bugFound.length > 1) body.appendChild(el('p', 'result__title', inc.title));
                if (inc.synthetic) body.appendChild(el('p', 'tag tag--synthetic', 'SYNTHETIC TEST: not a game bug'));
                body.appendChild(el('p', 'result__desc', inc.description));
                const replay = replayInfo(inc);
                const reports = reportsInfo(inc);
                const list = el('ul', 'checklist');
                list.append(
                    checkItem('ok', 'Testing paused', 'Before the game moved on, at the frame the rule broke'),
                    checkItem('ok', `Incident created: ${inc.incident_id}`, `Captured ${shortTime(inc.created_utc)}`),
                    checkItem('ok', 'Evidence captured', 'Trigger frame, movement history and every button press'),
                    checkItem(replay.kind, `Replay check: ${replay.label}`, replay.text),
                    checkItem(reports.kind, 'Reports', reports.text),
                );
                body.appendChild(list);
                const facts = el('dl', 'facts');
                const loc = inc.location || {};
                for (const [k, v] of [['Where', `world x ${loc.x}, y ${loc.y}`], ['Severity', inc.severity],
                                      ['Confidence', inc.confidence], ['Seen', `${inc.occurrences}×`]]) {
                    facts.append(el('dt', null, k), el('dd', null, String(v)));
                }
                body.appendChild(facts);
                body.appendChild(incidentLinks(inc, true));
                if (i === state.bugFound.length - 1) {
                    body.appendChild(el('p', 'result__next', 'Review the evidence, then press Resume testing to continue from this moment.'));
                }
                parts.push(body);
                return parts;
            }));
            card.hidden = false;
            return;
        }
        if (p === 'clean' || p === 'ended') {
            const r = state.runResult;
            const report = r.report;
            const pass = p === 'clean';
            card.className = `card result ${pass ? 'result--ok' : 'result--neutral'}`;
            const head = el('div', 'result__head');
            const ic = el('span', 'result__icon');
            ic.appendChild(icon(pass ? 'check' : 'flag'));
            head.appendChild(ic);
            const tx = el('div');
            tx.appendChild(el('p', 'result__kicker', pass ? 'Level complete · no bugs found' : 'Run ended'));
            tx.appendChild(el('p', 'result__title', pass ? 'Clean game tested successfully'
                : (RUN_TITLES[r.end_reason] || 'The run ended') + (report ? ` · ${report.headline}` : '')));
            head.appendChild(tx);
            const body = el('div', 'result__body');
            const d = detectorCount();
            const list = el('ul', 'checklist');
            if (pass) {
                list.append(
                    checkItem('ok', 'Clean game tested', state.status.game_is_clean_baseline
                        ? 'Its files match the pinned clean baseline' : 'The untouched original level'),
                    checkItem('ok', 'Mario reached the castle', `After ${num(report.agent_steps)} agent steps`),
                    checkItem('ok', `${d ? `${d} detectors` : 'Every detector'} active; none fired`, 'Checked on every frame of the run'),
                    checkItem((report.available || []).includes('report.pdf') ? 'ok' : 'wait', 'Run report saved',
                              `${report.run_id}: PDF, Markdown, final frame, GIF`),
                );
            } else {
                list.append(
                    checkItem('ok', 'No detector fired', 'Before the run ended'),
                    checkItem('idle', RUN_TITLES[r.end_reason] || 'The run ended',
                              report ? 'A run report was saved' : 'Run reports are written only for runs that reach the castle'),
                );
            }
            body.appendChild(list);
            if (report) body.appendChild(runLinks(report));
            if (pass) {
                body.appendChild(el('p', 'result__honest',
                    `"No bugs found" covers this run's path and these ${d || ''} checks only. It is not proof that the game has no bugs.`.replace('  ', ' ')));
            }
            body.appendChild(el('p', 'result__next', 'Start next run plays the level again; Reset clears the page.'));
            card.replaceChildren(head, body);
            card.hidden = false;
            return;
        }
        card.hidden = true;
        card.replaceChildren();
    }

    // ─── LIVE STATUS PANEL ───
    function renderTelemetry() {
        const t = state.telemetry;
        const mode = $('t-mode');
        const why = $('t-mode-why');
        const legacy = state.status.reward_mode === 'legacy_completion';
        if (!t) {
            mode.className = 'mode-chip mode-chip--none';
            mode.textContent = '—';
            why.textContent = 'Shown once testing starts';
        } else if (t.phase === 'explore') {
            mode.className = 'mode-chip mode-chip--explore';
            mode.textContent = 'Explore';
            why.textContent = 'Looking for places it has never visited';
        } else if (t.phase === 'complete') {
            mode.className = 'mode-chip mode-chip--complete';
            mode.textContent = 'Complete';
            why.textContent = 'Explored enough here; heading for the flag';
        } else {
            mode.className = 'mode-chip';
            mode.textContent = 'Play';
            why.textContent = legacy ? 'The 6M brain plays to finish the level' : 'Playing the level';
        }
        $('t-action').textContent = t ? t.action : '—';
        $('t-run').textContent = t ? `Run ${t.run} · step ${num(t.run_step)}` : '—';
        const prog = t && typeof t.progress === 'number' ? t.progress : null;
        $('t-progress-bar').style.width = prog === null ? '0' : `${Math.round(prog * 100)}%`;
        $('t-progress').textContent = prog === null ? '—' : `${Math.round(prog * 100)}% of the way to the castle (best this run)`;

        const cov = t && t.coverage;
        const o2 = state.facts && state.facts.objective2;
        const covEl = $('t-coverage');
        const covBar = $('t-coverage-bar');
        covEl.replaceChildren();
        if (cov) {
            const p = cov.covered / cov.total;
            covBar.style.width = `${(p * 100).toFixed(2)}%`;
            covEl.append(`${(p * 100).toFixed(2)}% of the reachable level`,
                         el('span', 'status-row__why', `${num(cov.new_since_start)} new pixels since the dashboard started · map from training`));
        } else if (!t && o2 && state.status.brain_approved && typeof o2.coverage_percent === 'number') {
            covBar.style.width = `${o2.coverage_percent}%`;
            covEl.append(`${o2.coverage_percent.toFixed(2)}% of the reachable level`,
                         el('span', 'status-row__why', 'From training; updates live while testing'));
        } else if (t && !cov) {
            covBar.style.width = '0';
            covEl.append('Not tracked for this brain');
        } else {
            covBar.style.width = '0';
            covEl.append('—');
        }
        const d = detectorCount();
        $('t-detectors').textContent = d ? `${d} active · checked every frame` : 'Active on every frame';
        const n = state.sessionIncidents.length;
        const seen = state.sessionIncidents.reduce((a, i) => a + (i.seen_this_session || 1), 0);
        $('t-bugs').textContent = n ? `${n} found${seen > n ? ` · ${seen} sightings` : ''}` : 'None yet';
    }

    // ─── BUG TRACKER (this session) ───
    function renderTracker() {
        const items = [];
        for (const inc of state.sessionIncidents) {
            const li = el('li');
            const b = el('button', `tracker-item${inc.synthetic ? ' tracker-item--synthetic' : ''}`);
            b.type = 'button';
            b.setAttribute('aria-label', `Open details of ${inc.title}`);
            const top = el('span', 'tracker-item__top');
            top.append(sevChip(inc.severity));
            if (inc.synthetic) top.append(el('span', 'tag tag--synthetic', 'synthetic test'));
            const replay = replayInfo(inc);
            top.append(el('span', `tag ${replay.kind === 'ok' ? 'tag--ok' : replay.kind === 'warn' ? 'tag--warn' : ''}`, `replay: ${replay.label.toLowerCase()}`));
            b.append(top, el('span', 'tracker-item__title', inc.title));
            const meta = el('span', 'tracker-item__meta');
            meta.append(el('span', null, shortTime(inc.created_utc)), el('span', null, `x ${inc.location.x}`),
                        el('span', null, `seen ${inc.occurrences}×`), el('span', null, inc.incident_id));
            b.append(meta);
            b.addEventListener('click', () => openIncident(inc.incident_id));
            li.append(b);
            items.push(li);
        }
        for (const report of [...state.sessionRuns].reverse()) {
            const li = el('li');
            const b = el('button', 'tracker-item tracker-item--run');
            b.type = 'button';
            const top = el('span', 'tracker-item__top');
            top.append(el('span', `tag ${report.bugs === 0 ? 'tag--ok' : 'tag--warn'}`, report.bugs === 0 ? 'no bugs found' : `${report.bugs} bug(s)`));
            b.append(top, el('span', 'tracker-item__title', `Run report: ${report.headline}`));
            const meta = el('span', 'tracker-item__meta');
            meta.append(el('span', null, shortTime(report.created_utc)), el('span', null, `${num(report.agent_steps)} steps`),
                        el('span', null, report.run_id));
            b.append(meta);
            b.addEventListener('click', () => openRun(report));
            li.append(b);
            items.push(li);
        }
        if (!items.length) {
            const started = state.steps > 0 || state.testing;
            const d = detectorCount();
            const li = el('li', `empty${started ? ' empty--ok' : ''}`);
            const ic = el('span', 'empty__icon');
            ic.appendChild(icon(started ? 'search' : 'ring'));
            li.append(ic, el('span', 'empty__title', started ? 'No bugs detected yet' : 'Nothing tested yet'),
                      document.createTextNode(started
                          ? `${d ? `${d} detectors are` : 'Every detector is'} checking every frame. A bug appears here the moment one fires.`
                          : 'Press Start testing. Every bug the AI finds appears here, with its evidence and reports.'));
            items.push(li);
        }
        bugList.replaceChildren(...items);
        renderTelemetry();
    }

    // ─── BRAIN CARD ───
    function renderBrain() {
        const s = state.status;
        const b = (state.facts && state.facts.brain) || {};
        const name = $('brain-name');
        const detail = $('brain-detail');
        const checks = $('brain-checks');
        checks.replaceChildren();
        if (!s.brain_path) {
            name.textContent = state.connected ? 'No trained brain found' : '—';
            detail.textContent = state.connected ? 'An untrained policy would play: install the final brain (README, step 4).' : '';
            return;
        }
        const steps = b.num_timesteps;
        name.textContent = s.brain_approved ? 'Final QA brain (Objective 2)'
            : s.reward_mode === 'legacy_completion' ? 'First brain (Objective 1)' : 'QA brain';
        detail.textContent = [steps ? `${num(steps)} training steps` : null,
                              b.parameters ? `${num(b.parameters)} parameters` : null,
                              s.brain_path].filter(Boolean).join(' · ');
        checks.append(s.brain_approved
            ? checkItem('ok', 'Verified', 'Its SHA-256 matches the approved Objective-2 record')
            : checkItem('warn', 'Not the approved final brain', 'The 16M brain is not installed or does not match its record'));
        checks.append(checkItem('ok', s.reward_mode === 'qa_exploration' ? 'Mode: QA exploration' : 'Mode: finish the level',
            s.reward_mode === 'qa_exploration' ? 'Explore first, then finish' : 'The Objective-1 reward'));
        if (s.game_variant === 'mario_clean') {
            checks.append(s.game_is_clean_baseline
                ? checkItem('ok', 'Clean game verified', 'Its files match the pinned baseline')
                : checkItem('warn', 'Clean game differs from its pin', 'Its files do not match the pinned baseline'));
        } else if (s.game_variant === 'mario_bugged') {
            const n = state.facts && state.facts.objective3 ? state.facts.objective3.benchmark_bugs.length : null;
            checks.append(checkItem('bug', 'Bugged copy loaded', `${n ? `${n} declared` : 'Declared'} benchmark bugs`));
        }
    }

    // ─── RENDER EVERYTHING THAT DEPENDS ON THE STATE ───
    function render() {
        const p = phase();
        renderPill(p);
        renderControls(p);
        renderNow(p);
        renderStepper(p);
        renderMonitor(p);
        renderResult(p);
        renderTelemetry();
        renderBrain();
        $('synthetic-badge').hidden = !((state.status.synthetic_probes || []).length);
    }

    // ─── LOG ───
    function note(message) {
        const ph = $('log-placeholder');
        if (ph) ph.remove();
        const p = el('p', 'log-note', message);
        logTerminal.appendChild(p);
        trimLog();
        logTerminal.scrollTop = logTerminal.scrollHeight;
    }

    function trimLog() {
        while (logTerminal.children.length > MAX_LOG_LINES) logTerminal.removeChild(logTerminal.firstChild);
    }

    function clearLog() {
        const p = el('p', 'placeholder', 'Nothing yet. The log fills in when testing starts.');
        p.id = 'log-placeholder';
        logTerminal.replaceChildren(p);
    }

    // ─── DATA FROM THE SERVER ───
    function rememberRun(report) {
        if (!report) return;
        const i = state.sessionRuns.findIndex((r) => r.run_id === report.run_id);
        if (i >= 0) state.sessionRuns[i] = report; else state.sessionRuns.push(report);
    }

    async function refreshStatus() {
        try {
            const res = await fetch('/api/status', { cache: 'no-store' });
            if (!res.ok) return;
            const s = await res.json();
            state.status = s;
            state.testing = !!s.testing;
            state.steps = s.steps || 0;
            state.pauseReason = s.pause_reason;
            state.bugFound = s.bug_found && s.bug_found.length ? s.bug_found : null;
            state.runResult = s.run_result || null;
            if (s.telemetry) state.telemetry = s.telemetry;
            if (state.runResult) rememberRun(state.runResult.report);
            render();
            renderTracker();
        } catch (err) {
            console.warn('could not load status', err);
        }
    }

    async function refreshIncidents() {
        try {
            const res = await fetch('/api/incidents', { cache: 'no-store' });
            if (!res.ok) return;
            state.sessionIncidents = (await res.json()).incidents || [];
            // Keep the Bug Found card current as its replay and reports finish.
            if (state.bugFound) {
                state.bugFound = state.bugFound.map((b) =>
                    state.sessionIncidents.find((i) => i.incident_id === b.incident_id) || b);
            }
            render();
            renderTracker();
        } catch (err) {
            console.warn('could not load incidents', err);
        }
    }

    let factsTimer = null;
    async function refreshFacts() {
        try {
            const res = await fetch('/api/project', { cache: 'no-store' });
            if (!res.ok) return;
            state.facts = await res.json();
            renderFacts();
            render();
            renderTracker();
            // The test count is collected in the background; ask again until known.
            clearTimeout(factsTimer);
            const tests = state.facts.engineering && state.facts.engineering.tests;
            if (tests && (tests.status === 'counting' || tests.status === 'idle')) {
                factsTimer = setTimeout(refreshFacts, 4000);
            }
        } catch (err) {
            console.warn('could not load the project facts', err);
        }
    }

    // ─── OVERVIEW + HOW IT WORKS: the project's facts ───
    function stat(label, value, sub, opts = {}) {
        const d = el('div', `stat${opts.pending ? ' stat--pending' : ''}`);
        d.append(el('p', 'stat__label', label), el('p', `stat__value${opts.ok ? ' stat__value--ok' : ''}`, value));
        if (sub) d.append(el('p', 'stat__sub', sub));
        return d;
    }

    function renderFacts() {
        const f = state.facts;
        if (!f) return;
        const o1 = f.objective1;
        const o2 = f.objective2;
        const o3 = f.objective3 || { detectors: [], benchmark_bugs: [], evidence: {} };
        const ev = o3.evidence || {};
        const eng = f.engineering || {};
        const brain = f.brain || {};
        const nDet = o3.detectors.length;
        const nBugs = o3.benchmark_bugs.length;

        setFact('detector-count', nDet ? String(nDet) : 'the');
        setFact('action-count', brain.actions ? String(brain.actions) : 'its');
        setFact('bug-count', nBugs ? String(nBugs) : 'planted');
        setFact('o1-steps', o1 && o1.timesteps ? num(o1.timesteps) : '—');
        setFact('o1-completion', o1 ? `Finishes the level in ${pct(o1.completion_rate)} of ${o1.episodes} test runs` : 'Result file not found');
        const covPct = o2 && typeof o2.coverage_percent === 'number' ? `${o2.coverage_percent.toFixed(2)}%` : null;
        setFact('o2-coverage', covPct || '—');
        setFact('o2-completion', o2
            ? `${num(o2.timesteps)} training steps · finishes ${pct(o2.completion_rate)} (was ${pct(o2.baseline_completion_rate)}) · ${o2.verdict}`
            : 'The final brain is not installed');
        const caught = (ev.benchmark_bugs_with_evidence || []).length;
        setFact('o3-caught', nBugs ? `${caught} of ${nBugs}` : '—');
        const cleanInc = ev.incidents_per_game ? ev.incidents_per_game.mario_clean : null;
        setFact('o3-clean', typeof cleanInc === 'number'
            ? `${nDet} detectors · ${cleanInc} incident${cleanInc === 1 ? '' : 's'} on the clean game · ${ev.clean_runs_no_bugs} clean "No Bugs Found" run reports`
            : `${nDet} detectors`);
        setFact('o1-flow', o1 ? `${num(o1.timesteps)} steps · finishes ${pct(o1.completion_rate)}` : '—');
        setFact('o2-flow', o2 ? `${num(o2.timesteps)} steps · ${covPct || '—'} of the reachable level · finishes ${pct(o2.completion_rate)} · ${o2.verdict}` : 'Not installed');

        const tiles = [];
        if (o2) tiles.push(stat('Training steps', num(o2.timesteps), o1 ? `Final QA brain, built on a ${num(o1.timesteps)}-step first brain` : 'Final QA brain'));
        if (brain.parameters) tiles.push(stat('Brain size', num(brain.parameters), `Learned parameters · ${brain.actions} actions`));
        if (covPct) tiles.push(stat('Reachable level explored', covPct, `${num(o2.covered)} of ${num(o2.reachable)} pixels`));
        if (o2) tiles.push(stat('Level completion', pct(o2.completion_rate), `${o2.completed} of ${o2.episodes} runs · was ${pct(o2.baseline_completion_rate)} · ${o2.verdict}`));
        if (nDet) {
            const eng5 = o3.detectors.filter((d) => d.group === 'engine').length;
            tiles.push(stat('Bug detectors', String(nDet), `${eng5} engine rules + ${nDet - eng5} collision and jump rules`));
        }
        if (nBugs) tiles.push(stat('Benchmark bugs', `${caught} of ${nBugs}`, 'Planted bugs with saved, replayed evidence'));
        if (typeof cleanInc === 'number') {
            tiles.push(stat('Incidents on the clean game', String(cleanInc), ev.clean_runs_no_bugs
                ? `Each would be a false alarm · ${ev.clean_runs_no_bugs} clean runs saved with "No Bugs Found"`
                : 'Each would be a false alarm · no clean-game run saved yet',
            { ok: cleanInc === 0 && ev.clean_runs_no_bugs > 0 }));
        }
        const tests = eng.tests || {};
        if (tests.status === 'done') tiles.push(stat('Automated tests', num(tests.count), 'Collected by pytest from the tests folder'));
        else if (tests.status === 'counting' || tests.status === 'idle') tiles.push(stat('Automated tests', 'Counting…', 'pytest is collecting them in the background', { pending: true }));
        if (eng.source_lines) tiles.push(stat('Project code', `${num(eng.source_lines)} lines`, `Python in ${eng.source_files} files, plus ${num(eng.test_lines)} lines of tests (game code not counted)`));
        if (eng.commits) tiles.push(stat('Commits', num(eng.commits), 'In the git history'));
        $('stat-grid').replaceChildren(...tiles);

        const cat = $('bug-catalogue');
        cat.replaceChildren(...o3.benchmark_bugs.map((b) => {
            const d = el('div', 'bug-catalogue__item');
            d.append(el('b', null, `${b.number}. ${b.name}`), el('span', null, b.summary));
            return d;
        }));

        const count = $('tab-history-count');
        count.hidden = !ev.incidents;
        count.textContent = ev.incidents ? String(ev.incidents) : '';
    }

    // ─── BUG HISTORY ───
    function table(headers, rows) {
        const t = el('table', 'data');
        const thead = el('thead');
        const tr = el('tr');
        for (const h of headers) tr.appendChild(el('th', null, h));
        thead.appendChild(tr);
        const tbody = el('tbody');
        rows.forEach((r) => tbody.appendChild(r));
        t.append(thead, tbody);
        return t;
    }

    function cell(...children) {
        const td = el('td');
        td.append(...children);
        return td;
    }

    async function refreshHistory() {
        try {
            const [incRes, runRes] = await Promise.all([
                fetch('/api/incidents?all=1', { cache: 'no-store' }),
                fetch('/api/runs', { cache: 'no-store' }),
            ]);
            const incidents = incRes.ok ? (await incRes.json()).incidents || [] : [];
            const runs = runRes.ok ? (await runRes.json()).runs || [] : [];
            renderHistory(incidents, runs);
        } catch (err) {
            console.warn('could not load the history', err);
        }
    }

    function renderHistory(incidents, runs) {
        const real = incidents.filter((i) => !i.synthetic);
        const per = (v) => real.filter((i) => i.game_variant === v).length;
        const cleanRuns = runs.filter((r) => r.game_variant === 'mario_clean');
        $('history-summary').replaceChildren(
            stat('Bug incidents saved', String(real.length), `${per('mario_bugged')} on the bugged game`),
            stat('Incidents on the clean game', String(per('mario_clean')), 'Each would be a false alarm',
                 { ok: per('mario_clean') === 0 && cleanRuns.length > 0 }),
            stat('Clean-run reports', String(runs.length), `${cleanRuns.filter((r) => r.bugs === 0).length} say "No Bugs Found"`),
        );
        const incBox = $('history-incidents');
        if (!incidents.length) {
            const e = el('div', 'empty');
            e.append(el('span', 'empty__title', 'No bug incidents saved yet'),
                     document.createTextNode('Test the bugged game on the Live Testing page; every bug found is saved here.'));
            incBox.replaceChildren(e);
        } else {
            incBox.replaceChildren(table(['When', 'Bug', 'Game', 'Severity', 'Replay', 'Seen', 'Evidence'],
                incidents.map((inc) => {
                    const tr = el('tr');
                    const title = el('span', 'data__title', inc.title);
                    const sub = el('span', 'data__sub', inc.incident_id);
                    const replay = replayInfo(inc);
                    const actions = el('div', 'actions');
                    const details = el('button', 'btn btn--secondary btn--sm', 'Details');
                    details.type = 'button';
                    details.addEventListener('click', () => openIncident(inc.incident_id));
                    actions.append(details);
                    const pdf = incidentFile(inc, 'report.pdf');
                    if (pdf) actions.append(link('PDF', fileUrl(inc.incident_id, pdf)));
                    const gif = incidentFile(inc, 'context.gif');
                    if (gif) actions.append(link('GIF', fileUrl(inc.incident_id, gif)));
                    actions.append(link('.zip', `/incidents/${encodeURIComponent(inc.incident_id)}/bundle.zip`, { download: true }));
                    actions.style.marginTop = '0';
                    tr.append(cell(el('span', 'data__nowrap', localTime(inc.created_utc))),
                              cell(title, sub, ...(inc.synthetic ? [el('span', 'tag tag--synthetic', 'synthetic test')] : [])),
                              cell(GAME_NAMES[inc.game_variant] || String(inc.game_variant)),
                              cell(sevChip(inc.severity)),
                              cell(el('span', `tag ${replay.kind === 'ok' ? 'tag--ok' : replay.kind === 'warn' ? 'tag--warn' : ''}`, replay.label)),
                              cell(`${inc.occurrences}×`), cell(actions));
                    return tr;
                })));
        }
        const runBox = $('history-runs');
        if (!runs.length) {
            const e = el('div', 'empty');
            e.append(el('span', 'empty__title', 'No run reports saved yet'),
                     document.createTextNode('A clean-game run that reaches the castle writes one.'));
            runBox.replaceChildren(e);
        } else {
            runBox.replaceChildren(table(['When', 'Result', 'Game', 'Steps', 'Report'],
                runs.map((r) => {
                    const tr = el('tr');
                    const actions = runLinks(r);
                    actions.style.marginTop = '0';
                    tr.append(cell(el('span', 'data__nowrap', localTime(r.created_utc))),
                              cell(el('span', 'data__title', r.headline), el('span', 'data__sub', r.run_id)),
                              cell(GAME_NAMES[r.game_variant] || String(r.game_variant || '—')),
                              cell(num(r.agent_steps)), cell(actions));
                    return tr;
                })));
        }
    }

    // ─── DIALOGS ───
    const incidentDialog = $('incident-dialog');
    const infoModal = $('info-modal');
    for (const dlg of [incidentDialog, infoModal]) {
        dlg.addEventListener('click', (e) => {
            if (e.target === dlg || e.target.closest('[data-close]')) dlg.close();
        });
    }
    $('info-btn').addEventListener('click', () => infoModal.showModal());

    function viewer(sources) {
        const box = el('div', 'incident-detail__viewer');
        const img = el('img');
        const caption = el('p', 'incident-detail__caption');
        const bar = el('div', 'evidence-switch');
        const avail = sources.filter((s) => s.url);
        const show = (s) => {
            img.src = s.url;
            img.alt = s.alt;
            caption.textContent = s.caption;
            bar.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === s.key)));
        };
        for (const s of sources) {
            const b = el('button', 'btn btn--secondary btn--sm', s.url ? s.label : `${s.label} (not ready)`);
            b.type = 'button';
            b.dataset.key = s.key;
            b.disabled = !s.url;
            b.addEventListener('click', () => show(s));
            bar.appendChild(b);
        }
        box.append(img, bar, caption);
        if (avail.length) show(avail[0]);
        else caption.textContent = 'No picture has been saved for this yet.';
        return box;
    }

    async function openIncident(id) {
        let detail;
        try {
            const res = await fetch(`/api/incidents/${encodeURIComponent(id)}`, { cache: 'no-store' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            detail = await res.json();
        } catch (err) {
            note(`— could not open ${id}: ${err.message} —`);
            return;
        }
        const inc = detail.summary;
        const record = detail.record || {};
        const kicker = $('incident-dialog-kicker');
        kicker.replaceChildren(sevChip(inc.severity), el('span', null, inc.incident_id),
                               el('span', null, GAME_NAMES[inc.game_variant] || inc.game_variant));
        if (inc.synthetic) kicker.append(el('span', 'tag tag--synthetic', 'synthetic test, not a game bug'));
        $('incident-dialog-title').textContent = inc.title;

        const gif = incidentFile(inc, 'context.gif');
        const grid = el('div', 'incident-detail');
        grid.append(viewer([
            { key: 'gif', label: 'GIF', url: gif ? fileUrl(id, gif) : null, alt: 'The moments leading up to the bug',
              caption: 'context.gif: the moments before the bug; its last frame is the trigger frame.' },
            { key: 'trigger', label: 'Trigger frame', url: incidentFile(inc, 'trigger.png') ? fileUrl(id, 'trigger.png') : null,
              alt: 'The exact frame the bug was detected on', caption: 'trigger.png: the exact frame, full resolution.' },
        ]));
        const right = el('div');
        right.append(el('p', 'incident-detail__desc', inc.description));
        const cls = record.classification || {};
        const conf = cls.detector_confidence || {};
        const replay = replayInfo(inc);
        const reproduction = detail.reproduction || {};
        right.append(el('h3', null, 'What was detected'));
        const facts = el('dl', 'facts');
        const add = (k, v, explain) => {
            const dd = el('dd', null, v);
            if (explain) dd.appendChild(el('span', 'explain', explain));
            facts.append(el('dt', null, k), dd);
        };
        add('Rule', (record.detector && record.detector.id) || inc.category, record.detector && record.detector.message);
        add('Where', `world x ${inc.location.x}, y ${inc.location.y}`);
        add('When', localTime(inc.created_utc), `Episode ${inc.episode_index}, agent step ${inc.episode_agent_step}`);
        add('Severity', inc.severity, cls.severity_rationale);
        add('Confidence', inc.confidence, (conf.reasons || [])[0]);
        add('Replay', replay.label, reproduction.detail || replay.text);
        add('Seen', `${inc.occurrences}×`, 'Repeat sightings raise this count; no duplicate incident is created');
        add('Brain', inc.brain_approved ? 'Approved final QA brain' : 'Not the approved Objective-2 brain');
        right.append(facts);
        right.append(el('h3', null, 'Reports and evidence'));
        const links = incidentLinks(inc, false);
        links.append(link('Record (JSON)', `/api/incidents/${encodeURIComponent(id)}`));
        right.append(links);
        grid.append(right);
        $('incident-dialog-body').replaceChildren(grid);
        if (!incidentDialog.open) incidentDialog.showModal();
    }

    function openRun(report) {
        const kicker = $('incident-dialog-kicker');
        kicker.replaceChildren(el('span', `tag ${report.bugs === 0 ? 'tag--ok' : 'tag--warn'}`, report.bugs === 0 ? 'no bugs found' : `${report.bugs} bug(s)`),
                               el('span', null, report.run_id));
        $('incident-dialog-title').textContent = `Run report: ${report.headline}`;
        const has = (n) => (report.available || []).includes(n);
        const grid = el('div', 'incident-detail');
        grid.append(viewer([
            { key: 'gif', label: 'GIF', url: has('finish.gif') ? runFileUrl(report.run_id, 'finish.gif') : null,
              alt: 'The last seconds of the run', caption: 'finish.gif: the last seconds of the run.' },
            { key: 'final', label: 'Final frame', url: has('final.png') ? runFileUrl(report.run_id, 'final.png') : null,
              alt: 'The last frame of the run', caption: 'final.png: the last frame, full resolution.' },
        ]));
        const right = el('div');
        right.append(el('p', 'incident-detail__desc', `${report.outcome} after ${num(report.agent_steps)} agent steps.`));
        const facts = el('dl', 'facts');
        const VERDICTS = { no_bugs_found: 'No bugs found', bugs_found: 'Bugs found' };
        facts.append(el('dt', null, 'Verdict'), el('dd', null, VERDICTS[report.verdict] || report.headline),
                     el('dt', null, 'When'), el('dd', null, localTime(report.created_utc)),
                     el('dt', null, 'Bugs'), el('dd', null, String(report.bugs)));
        right.append(el('h3', null, 'The run'), facts,
                     el('p', 'result__honest', '"No bugs found" covers this run and these detectors only; it is not proof that the game has no bugs.'),
                     el('h3', null, 'Reports and evidence'), runLinks(report));
        grid.append(right);
        $('incident-dialog-body').replaceChildren(grid);
        if (!incidentDialog.open) incidentDialog.showModal();
    }

    // ─── CONTROLS ───
    startBtn.addEventListener('click', () => {
        state.testing = true;
        state.pauseReason = null;
        state.bugFound = null;       // the server clears bug_found on resume, and says so
        state.runResult = null;      // ...and run_result: Start plays the next run
        state.evidenceChoice = 'auto';
        socket.emit('start_testing');
        const ph = $('log-placeholder');
        if (ph) ph.remove();
        render();
        renderTracker();
    });

    stopBtn.addEventListener('click', () => {
        socket.emit('stop_testing');
        state.testing = false;
        state.pauseReason = 'stop';
        render();
    });

    function resetDashboard() {
        socket.emit('stop_testing');
        socket.emit('reset_game');
        // Reset ends the session: the log and the Bug Tracker start empty
        // (the server starts a new session list; every incident stays saved).
        state.testing = false;
        state.steps = 0;
        state.pauseReason = 'reset';
        state.bugFound = null;
        state.runResult = null;
        state.telemetry = null;
        state.sessionIncidents = [];
        state.sessionRuns = [];
        state.evidenceChoice = 'auto';
        clearLog();
        clearFrame();
        render();
        renderTracker();
    }
    resetBtn.addEventListener('click', resetDashboard);

    // Picking the other game resets the dashboard, exactly like Reset, and
    // the server loads that game; START TESTING then plays it.
    for (const radio of gameRadios) {
        radio.addEventListener('change', () => {
            if (!radio.checked || radio.value === state.status.game_variant) return;
            resetDashboard();
            state.switching = true;
            state.status = { ...state.status, game_variant: radio.value };
            render();
            socket.emit('switch_game', { variant: radio.value });
        });
    }

    // ─── SOCKET EVENTS ───
    socket.on('game_switched', (data) => {
        state.switching = false;
        if (!(data && data.ok)) {
            note(`— could not switch the game: ${(data && data.error) || 'unknown error'} —`);
        }
        refreshStatus();         // the server's selection wins, whichever it is
        refreshFacts();
    });

    socket.on('video_frame', (data) => {
        if (!state.testing) return;
        // The JPEG arrives as raw binary (an ArrayBuffer). An ArrayBuffer is
        // always truthy even when empty, so check byteLength.
        if (data.frame && data.frame.byteLength > 0) {
            const hadFrame = videoFeed.hasAttribute('src');
            const url = URL.createObjectURL(new Blob([data.frame], { type: 'image/jpeg' }));
            videoFeed.src = url;
            if (currentFrameUrl) URL.revokeObjectURL(currentFrameUrl);
            currentFrameUrl = url;
            if (!hadFrame) renderMonitor(phase());
        }
    });

    socket.on('agent_log', (data) => {
        if (!state.testing) return;
        // The detector's own line ("Step N: 🚨 BUG FOUND: ...") stays in the
        // log; the Bug Tracker lists what the server actually recorded.
        const p = el('p', data.log && data.log.includes('🚨 BUG FOUND:') ? 'log-bug' : null, data.log);
        logTerminal.appendChild(p);
        trimLog();
        logTerminal.scrollTop = logTerminal.scrollHeight;
    });

    socket.on('telemetry', (t) => {
        if (!t) return;
        state.telemetry = t;
        state.steps = t.session_steps || state.steps;
        renderTelemetry();
        if (state.testing) renderNow(phase());
    });

    // A pause the SERVER initiated - most often the user closing the game
    // window with its X. The session is kept, so START TESTING reopens the
    // window and carries on from the same moment.
    const PAUSE_NOTES = {
        window_closed: '— game window closed: testing paused. Press Resume testing to reopen it and continue —',
        session_ended: '— the agent session ended —',
        error: '— testing paused after an error (see the server console) —',
        bug_found: '— BUG FOUND: testing stopped. The evidence is saved; press Resume testing to continue —',
        capture_failed: '— an anomaly was detected but its evidence could not be saved (see the server console) —',
        level_complete: '— LEVEL COMPLETE: testing stopped. Start next run plays again; Reset clears the page —',
        mario_died: '— MARIO DIED: testing stopped. Start next run plays again; Reset clears the page —',
        run_ended: '— the run ended: testing stopped. Start next run plays again; Reset clears the page —',
    };
    socket.on('testing_paused', (data) => {
        const reason = (data && data.reason) || '';
        note(PAUSE_NOTES[reason] || '— testing paused —');
        state.testing = false;
        state.pauseReason = reason;
        render();
    });

    socket.on('bug_found', (data) => {
        state.bugFound = (data && data.incidents) || [];
        state.testing = false;
        state.pauseReason = 'bug_found';
        state.evidenceChoice = 'auto';
        render();
        refreshIncidents();
        refreshFacts();
        if ($('view-live').hidden) showView('live');
    });
    socket.on('bug_cleared', () => { state.bugFound = null; render(); });
    socket.on('run_finished', (result) => {
        if (!result) return;
        state.runResult = result;
        state.testing = false;
        state.evidenceChoice = 'auto';
        rememberRun(result.report);
        render();
        renderTracker();
        refreshFacts();
    });
    socket.on('run_cleared', () => { state.runResult = null; render(); });
    socket.on('run_report_updated', (report) => {     // its GIF/MD/PDF finished rendering
        if (!report || !state.sessionRuns.some((r) => r.run_id === report.run_id)) return;
        rememberRun(report);
        if (state.runResult && state.runResult.report && state.runResult.report.run_id === report.run_id) {
            state.runResult = { ...state.runResult, report };
        }
        render();
        renderTracker();
    });
    socket.on('incident_updated', () => {              // a report finished rendering
        refreshIncidents();
        if (!$('view-history').hidden) refreshHistory();
    });
    socket.on('incident_occurrence', (inc) => {
        note(`— seen again: ${inc.title} (${inc.incident_id}), now ${inc.occurrences}× —`);
        refreshIncidents();
    });
    socket.on('incident_capture_failed', (data) => {
        note(`— anomaly detected, but its evidence could not be saved: ${(data && data.error) || 'unknown error'} —`);
    });

    // ─── CONNECTION LIFECYCLE ───
    // The server pauses whenever a client connects or disconnects (keeping
    // the session and the game window), so after any reconnect it is
    // definitively NOT running - the page has to agree. Resume then continues
    // the same session.
    socket.on('disconnect', () => {
        if (state.testing) note('— connection lost, testing stopped —');
        state.connected = false;
        state.testing = false;
        render();
    });

    socket.on('connect', () => {
        const reconnect = state.everConnected;
        state.connected = true;
        state.everConnected = true;
        if (reconnect) {
            // A dropped connection came back: nothing was reset, so the same
            // session resumes on START TESTING.
            note('— reconnected, press Resume testing to continue —');
            clearFrame();
            refreshStatus();
            refreshIncidents();
            return;
        }
        // A freshly loaded page (a refresh, a reopened tab) starts from a clean
        // dashboard, exactly like Reset. The server answers once the reset has
        // run; only then is its state read, so no old banner, report or bug is
        // shown again.
        render();
        socket.emit('page_opened', () => {
            refreshStatus();
            refreshIncidents();
            refreshFacts();
        });
    });

    // The legend's state labels carry the same icons as the live ones.
    document.querySelectorAll('.legend .pill').forEach((pill) => {
        const kind = ['info', 'bug', 'ok', 'warn'].find((k) => pill.classList.contains(`pill--${k}`));
        const box = pill.querySelector('.pill__icon');
        box.replaceChildren(kind === 'info' ? el('span', 'live-dot')
            : icon({ bug: 'bug', ok: 'check', warn: 'alert' }[kind] || 'pause'));
    });
    showView((location.hash || '#overview').slice(1));
    render();
    renderTracker();
});
