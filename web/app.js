const { Room, RoomEvent, Track, ConnectionState } = LivekitClient;

const els = {
  start: document.getElementById("start-btn"),
  enableAudio: document.getElementById("enable-audio-btn"),
  mute: document.getElementById("mute-btn"),
  end: document.getElementById("end-btn"),
  quit: document.getElementById("quit-btn"),
  title: document.getElementById("status-title"),
  detail: document.getElementById("status-detail"),
};

const state = {
  room: null,
  audioElements: new Map(),
  agentJoined: false,
  agentTimer: null,
  starting: false,
  muted: false,
  cleaning: false,
};

function setStatus(title, detail, mode = "idle") {
  els.title.textContent = title;
  els.detail.textContent = detail;
  document.body.dataset.mode = mode;
}

function setConnectedControls() {
  els.start.hidden = true;
  els.mute.hidden = false;
  els.end.hidden = false;
  els.enableAudio.hidden = true;
}

function resetControls() {
  els.start.hidden = false;
  els.start.disabled = false;
  els.mute.hidden = true;
  els.end.hidden = true;
  els.enableAudio.hidden = true;
  els.mute.textContent = "Mute";
  els.mute.classList.remove("active");
}

function clearAgentTimer() {
  if (state.agentTimer !== null) {
    window.clearTimeout(state.agentTimer);
    state.agentTimer = null;
  }
}

function markAgentJoined() {
  if (state.agentJoined) {
    return;
  }
  state.agentJoined = true;
  clearAgentTimer();
  setStatus("Listening", "COSMION is online. Say “Hello, COSMION.”", "listening");
}

function friendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);
  if (/permission|notallowed|denied/i.test(message)) {
    return "Microphone permission is required. Allow it in your browser and try again.";
  }
  if (/token|launcher|fetch|network|room/i.test(message)) {
    return "The local launcher could not start the room. Check the COSMION window for details.";
  }
  return message;
}

async function cleanupRoom() {
  if (state.cleaning) {
    return;
  }
  state.cleaning = true;
  clearAgentTimer();
  const room = state.room;
  state.room = null;
  if (room) {
    try {
      await room.disconnect();
    } catch (error) {
      console.warn("Room disconnect failed", error);
    }
  }
  state.cleaning = false;
}

async function resetAfterDisconnect(title, detail) {
  clearAgentTimer();
  for (const element of state.audioElements.values()) {
    element.remove();
  }
  state.audioElements.clear();
  state.agentJoined = false;
  state.muted = false;
  resetControls();
  setStatus(title, detail, "idle");
}

function wireRoom(room) {
  room.on(RoomEvent.TrackSubscribed, (track, publication) => {
    const isAudio = track.kind === Track.Kind.Audio || track.kind === "audio";
    if (!isAudio) {
      return;
    }

    const element = track.attach();
    element.autoplay = true;
    element.playsInline = true;
    element.style.display = "none";
    document.body.appendChild(element);
    state.audioElements.set(track.sid || publication.trackSid, element);
    markAgentJoined();

    element.play().catch(() => {
      els.enableAudio.hidden = false;
      setStatus("Audio paused", "Your browser blocked playback. Click Enable Audio.", "error");
    });
  });

  room.on(RoomEvent.TrackUnsubscribed, (track, publication) => {
    const key = track.sid || publication.trackSid;
    const element = state.audioElements.get(key);
    if (element) {
      element.remove();
      state.audioElements.delete(key);
    }
  });

  room.on(RoomEvent.ParticipantConnected, (participant) => {
    if (participant.identity !== room.localParticipant.identity) {
      markAgentJoined();
    }
  });

  room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
    if (!state.agentJoined) {
      return;
    }
    const agentSpeaking = speakers.some(
      (speaker) => speaker.identity !== room.localParticipant.identity,
    );
    if (agentSpeaking) {
      setStatus("COSMION is speaking", "Listening to the reply…", "speaking");
    } else {
      setStatus("Listening", "Go ahead, sir.", "listening");
    }
  });

  room.on(RoomEvent.ConnectionStateChanged, (connectionState) => {
    if (connectionState === ConnectionState.Connected && state.agentJoined) {
      setStatus("Listening", "Go ahead, sir.", "listening");
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    if (state.room === room) {
      state.room = null;
      void resetAfterDisconnect(
        "Disconnected",
        "Start a new conversation when you are ready.",
      );
    }
  });
}

async function startConversation() {
  if (state.starting || state.room) {
    return;
  }

  state.starting = true;
  els.start.disabled = true;
  els.enableAudio.hidden = true;
  setStatus("Connecting", "Requesting a local room token…", "connecting");

  try {
    const response = await fetch("/token", {
      headers: { "X-Cosmion": "1" },
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Local launcher returned ${response.status}`);
    }
    const session = await response.json();

    const room = new Room({ adaptiveStream: false, dynacast: false });
    state.room = room;
    wireRoom(room);

    await room.connect(session.url, session.token);
    try {
      await room.startAudio();
    } catch (error) {
      console.warn("Audio playback needs a click", error);
      els.enableAudio.hidden = false;
    }

    try {
      await room.localParticipant.setMicrophoneEnabled(true);
    } catch (error) {
      throw new Error("Microphone permission is required. Allow it in your browser and try again.");
    }

    state.muted = false;
    setConnectedControls();
    setStatus("Connecting", "Waiting for COSMION to join…", "connecting");

    if (room.remoteParticipants.size > 0) {
      markAgentJoined();
    } else {
      state.agentTimer = window.setTimeout(() => {
        if (!state.agentJoined) {
          setStatus(
            "Waiting for COSMION",
            "The room is connected, but the worker has not joined. Check the COSMION window.",
            "error",
          );
        }
      }, 25000);
    }
  } catch (error) {
    console.error(error);
    setStatus("Could not start", friendlyError(error), "error");
    await cleanupRoom();
    resetControls();
  } finally {
    state.starting = false;
    els.start.disabled = false;
  }
}

async function endConversation() {
  await cleanupRoom();
  await resetAfterDisconnect("Ready", "Start a conversation and speak naturally.");
}

async function toggleMute() {
  const room = state.room;
  if (!room) {
    return;
  }

  const enableMicrophone = state.muted;
  try {
    await room.localParticipant.setMicrophoneEnabled(enableMicrophone);
    state.muted = !enableMicrophone;
    els.mute.textContent = state.muted ? "Unmute" : "Mute";
    els.mute.classList.toggle("active", state.muted);
    if (state.muted) {
      setStatus("Muted", "Unmute when you are ready to speak.", "idle");
    } else {
      setStatus("Listening", "Go ahead, sir.", "listening");
    }
  } catch (error) {
    setStatus("Microphone error", friendlyError(error), "error");
  }
}

async function quitCosmion() {
  const confirmed = window.confirm("Stop COSMION and close the local launcher?");
  if (!confirmed) {
    return;
  }

  els.start.disabled = true;
  els.quit.disabled = true;
  setStatus("Stopping", "Shutting down COSMION…", "connecting");
  await cleanupRoom();
  try {
    await fetch("/shutdown", {
      method: "POST",
      headers: { "X-Cosmion": "1" },
    });
  } catch (error) {
    console.warn("Shutdown request failed", error);
  }
  setStatus("COSMION stopped", "You can close this tab.", "idle");
}

async function checkLauncher() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Health check returned ${response.status}`);
    }
  } catch (error) {
    setStatus(
      "Launcher unavailable",
      "The local COSMION server is not responding. Close this tab and start COSMION again.",
      "error",
    );
  }
}

els.start.addEventListener("click", startConversation);
els.end.addEventListener("click", endConversation);
els.mute.addEventListener("click", toggleMute);
els.quit.addEventListener("click", quitCosmion);
els.enableAudio.addEventListener("click", async () => {
  if (!state.room) {
    return;
  }
  try {
    await state.room.startAudio();
    els.enableAudio.hidden = true;
    setStatus("Listening", "Go ahead, sir.", "listening");
  } catch (error) {
    setStatus("Audio blocked", "Click the page, then try Enable Audio again.", "error");
  }
});

window.addEventListener("beforeunload", () => {
  if (state.room) {
    void state.room.disconnect();
  }
});

void checkLauncher();
els.start.focus();
