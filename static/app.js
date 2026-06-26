const API_ENDPOINT = "/api/chat";
const STORAGE_KEY = "shef.chats.v1";
const PRIVACY_ACCEPTANCE_KEY = "shef.privacy.accepted.v1";
const USAGE_LOGGING_ENABLED = window.SHEF_CONFIG?.usageLoggingEnabled === true;
const MAX_CHATS = 10;
const TARGET_AUDIO_SAMPLE_RATE = 16000;
const RESPONSE_MODE_AUTO = "auto";
const RESPONSE_MODE_RECIPE_OPTIONS = "recipe_options";
const RESPONSE_MODE_FULL_RECIPE = "full_recipe";
const WELCOME_TEXT =
  "Hi, I am Shef. Send ingredients by text, image, or voice and I will help you turn them into a recipe.";
const MAX_MESSAGE_CHARS = 2000;
const CLIENT_BLOCKED_PHRASES = [
  "ignore previous instructions",
  "ignore all previous instructions",
  "reveal system prompt",
  "reveal your system prompt",
  "show system prompt",
  "show your system prompt",
  "developer message",
  "system message",
  "api key",
  "secret key",
  "password",
  "bypass",
  "pretend you are",
  "act as",
  "disregard",
  "override",
  "jailbreak",
  "do anything now",
];

const composerForm = document.querySelector("#composerForm");
const messageInput = document.querySelector("#messageInput");
const messageList = document.querySelector("#messageList");
const addButton = document.querySelector("#addButton");
const imageInput = document.querySelector("#imageInput");
const attachmentTray = document.querySelector("#attachmentTray");
const micButton = document.querySelector("#micButton");
const recordingBar = document.querySelector("#recordingBar");
const recordingTimer = document.querySelector("#recordingTimer");
const stopRecordingButton = document.querySelector("#stopRecordingButton");
const composerError = document.querySelector("#composerError");
const newChatButton = document.querySelector("#newChatButton");
const sidebar = document.querySelector("#sidebar");
const menuButton = document.querySelector("#menuButton");
const sidebarCloseButton = document.querySelector("#sidebarCloseButton");
const sidebarBackdrop = document.querySelector("#sidebarBackdrop");
const chatList = document.querySelector("#chatList");
const sendButton = document.querySelector("#sendButton");
const toastContainer = document.querySelector("#toastContainer");
const appShell = document.querySelector(".app-shell");
const privacyButton = document.querySelector("#privacyButton");
const privacyModal = document.querySelector("#privacyModal");
const privacyIntro = document.querySelector("#privacyIntro");
const privacyRetention = document.querySelector("#privacyRetention");
const privacyUseTerms = document.querySelector("#privacyUseTerms");
const privacyCloseButton = document.querySelector("#privacyCloseButton");
const privacyAcceptanceCheckboxes = Array.from(
  document.querySelectorAll(".privacy-acceptance-checkbox")
);
const privacyAcceptButton = document.querySelector("#privacyAcceptButton");
const privacyAcceptanceUsageText = document.querySelector("#privacyAcceptanceUsageText");
const privacyAcceptanceTermsText = document.querySelector("#privacyAcceptanceTermsText");

let state = loadState();
let imageAttachment = null;
let audioAttachment = null;
let recorder = null;
let recordingStartedAt = 0;
let recordingInterval = null;
let typingNode = null;
let isSending = false;
let recipeOptionPreview = null;
let privacyConfirmationRequired = false;

const createId = () =>
  window.crypto?.randomUUID ? window.crypto.randomUUID() : `${Date.now()}-${Math.random()}`;

const nowIso = () => new Date().toISOString();

const formatClock = (date = new Date()) =>
  date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

const formatChatTime = (value) => {
  const date = new Date(value);
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  return sameDay
    ? formatClock(date)
    : date.toLocaleDateString([], { month: "short", day: "numeric" });
};

const formatDuration = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
};

const fileSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const fileToDataUrl = (fileOrBlob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(fileOrBlob);
  });

const INLINE_MARKDOWN_PATTERN =
  /(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;

const RECIPE_SECTION_LABELS =
  /^(Recipe Name|Recipe|Ingredients|Instructions|Steps|Method|Tips|Notes|Substitutions|Pantry Items|Optional Pantry Items|Recipe Options|Option\s+\d+)\s*:\s*(.*)$/i;
const RECIPE_OPTION_TITLE_PATTERN = /^\s*(\d+)[.)]\s+(.+)$/;
const RECIPE_OPTION_DESCRIPTION_PATTERN =
  /^\s*(?:[-*\u2022]\s*)?(?:\*\*)?(Flavor\s*(?:&|and)\s*Texture|Occasion\s*(?:&|and)\s*Fit|Pro\s*Tip)(?:\*\*)?\s*(?:\u2014|\u2013|-|:)\s*(.+)$/i;

const stripMarkdownDecorators = (value) =>
  String(value ?? "")
    .replace(/^#{1,6}\s+/, "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`/g, "")
    .trim();

const normaliseDescriptionKey = (label) =>
  stripMarkdownDecorators(label)
    .toLowerCase()
    .replace(/[^a-z]/g, "");

const descriptionKeyForLabel = (label) => {
  const key = normaliseDescriptionKey(label);
  if (key.includes("flavor") && key.includes("texture")) return "flavorTexture";
  if (key.includes("occasion") && key.includes("fit")) return "occasionFit";
  if (key.includes("pro") && key.includes("tip")) return "proTip";
  return "";
};

const parseRecipeOptions = (text) => {
  const lines = String(text ?? "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const options = [];
  let current = null;

  const commitCurrent = () => {
    if (!current?.title) return;
    options.push({
      title: current.title,
      descriptions: {
        flavorTexture: current.descriptions.flavorTexture || "",
        occasionFit: current.descriptions.occasionFit || "",
        proTip: current.descriptions.proTip || "",
      },
    });
  };

  lines.forEach((line) => {
    if (/^recipe options\s*:?\s*$/i.test(stripMarkdownDecorators(line))) {
      return;
    }

    const titleMatch = line.match(RECIPE_OPTION_TITLE_PATTERN);
    if (titleMatch) {
      commitCurrent();
      current = {
        title: stripMarkdownDecorators(titleMatch[2]).replace(/[:.]\s*$/, ""),
        descriptions: {},
      };
      return;
    }

    const descriptionMatch = line.match(RECIPE_OPTION_DESCRIPTION_PATTERN);
    if (current && descriptionMatch) {
      const descriptionKey = descriptionKeyForLabel(descriptionMatch[1]);
      if (descriptionKey) {
        current.descriptions[descriptionKey] = stripMarkdownDecorators(descriptionMatch[2]);
      }
    }
  });

  commitCurrent();

  const uniqueTitles = new Set();
  return options
    .filter((option) => {
      const key = option.title.toLowerCase();
      if (!option.title || uniqueTitles.has(key)) return false;
      uniqueTitles.add(key);
      return true;
    })
    .slice(0, 5);
};

const appendInlineMarkdown = (parent, value) => {
  const text = String(value ?? "");
  let cursor = 0;

  for (const match of text.matchAll(INLINE_MARKDOWN_PATTERN)) {
    const token = match[0];
    const start = match.index ?? 0;

    if (start > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, start)));
    }

    let tagName = "";
    let innerText = token;

    if (token.startsWith("**") && token.endsWith("**")) {
      tagName = "strong";
      innerText = token.slice(2, -2);
    } else if (token.startsWith("__") && token.endsWith("__")) {
      tagName = "strong";
      innerText = token.slice(2, -2);
    } else if (token.startsWith("`") && token.endsWith("`")) {
      tagName = "code";
      innerText = token.slice(1, -1);
    } else if (
      (token.startsWith("*") && token.endsWith("*")) ||
      (token.startsWith("_") && token.endsWith("_"))
    ) {
      tagName = "em";
      innerText = token.slice(1, -1);
    }

    if (tagName) {
      const inlineNode = document.createElement(tagName);
      inlineNode.textContent = innerText;
      parent.append(inlineNode);
    } else {
      parent.append(document.createTextNode(token));
    }

    cursor = start + token.length;
  }

  if (cursor < text.length) {
    parent.append(document.createTextNode(text.slice(cursor)));
  }
};

const isListLine = (line) => /^\s*(?:[-*\u2022]|\d+[.)])\s+/.test(line);

const parseHeadingLine = (line) => {
  const markdownHeading = line.match(/^\s{0,3}#{1,6}\s+(.+)$/);
  if (markdownHeading) {
    return { heading: stripMarkdownDecorators(markdownHeading[1]), trailing: "" };
  }

  const boldHeading = line.match(/^\s*\*\*(.+?)\*\*:?\s*$/);
  if (boldHeading) {
    return { heading: stripMarkdownDecorators(boldHeading[1]), trailing: "" };
  }

  const recipeLabel = line.match(RECIPE_SECTION_LABELS);
  if (!recipeLabel) return null;

  const label = recipeLabel[1].replace(/\s+/g, " ").trim();
  const trailing = recipeLabel[2].trim();
  if (/^recipe name$/i.test(label) && trailing) {
    return { heading: stripMarkdownDecorators(trailing), trailing: "" };
  }

  return { heading: stripMarkdownDecorators(label), trailing };
};

const appendParagraph = (parent, lines) => {
  const paragraphText = lines.join(" ").replace(/\s+/g, " ").trim();
  if (!paragraphText) return;

  const paragraph = document.createElement("p");
  appendInlineMarkdown(paragraph, paragraphText);
  parent.append(paragraph);
};

const renderAssistantText = (text) => {
  const container = document.createElement("div");
  container.className = "assistant-content";

  const lines = String(text ?? "")
    .replace(/\r\n?/g, "\n")
    .split("\n");

  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();

    if (!line) {
      index += 1;
      continue;
    }

    const headingLine = parseHeadingLine(line);
    if (headingLine) {
      if (headingLine.heading) {
        const heading = document.createElement("h3");
        appendInlineMarkdown(heading, headingLine.heading);
        container.append(heading);
      }
      if (headingLine.trailing) {
        appendParagraph(container, [headingLine.trailing]);
      }
      index += 1;
      continue;
    }

    if (isListLine(line)) {
      const isOrdered = /^\s*\d+[.)]\s+/.test(line);
      const list = document.createElement(isOrdered ? "ol" : "ul");

      while (index < lines.length) {
        const itemLine = lines[index].trim();
        if (!itemLine || !isListLine(itemLine)) break;
        if (/^\s*\d+[.)]\s+/.test(itemLine) !== isOrdered) break;

        const itemText = itemLine.replace(/^\s*(?:[-*\u2022]|\d+[.)])\s+/, "").trim();
        const item = document.createElement("li");
        appendInlineMarkdown(item, itemText);
        list.append(item);
        index += 1;
      }

      container.append(list);
      continue;
    }

    const paragraphLines = [];
    while (index < lines.length) {
      const paragraphLine = lines[index].trim();
      if (!paragraphLine || isListLine(paragraphLine) || parseHeadingLine(paragraphLine)) {
        break;
      }
      paragraphLines.push(paragraphLine);
      index += 1;
    }
    appendParagraph(container, paragraphLines);
  }

  if (!container.children.length) {
    appendParagraph(container, [text]);
  }

  return container;
};

const TOAST_ICONS = {
  warning: "⚠",
  error: "✕",
  info: "ℹ",
};

const TOAST_TITLES = {
  warning: "Input Blocked",
  error: "Request Failed",
  info: "Notice",
};

const showToast = (message, type = "warning", durationMs = 5000, title = "") => {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon" aria-hidden="true">${TOAST_ICONS[type] || TOAST_ICONS.warning}</span>
    <div class="toast-body">
      <div class="toast-title">${escapeHtml(title || TOAST_TITLES[type] || TOAST_TITLES.warning)}</div>
      <div class="toast-message">${escapeHtml(message)}</div>
    </div>
    <button class="toast-close" type="button" aria-label="Dismiss">&times;</button>
    <div class="toast-progress" style="animation-duration: ${durationMs}ms"></div>
  `;

  const dismiss = () => {
    if (toast.classList.contains("is-leaving")) return;
    toast.classList.add("is-leaving");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  };

  toast.querySelector(".toast-close").addEventListener("click", dismiss);
  const autoTimer = setTimeout(dismiss, durationMs);
  toast.addEventListener("mouseenter", () => clearTimeout(autoTimer));
  toast.addEventListener("mouseleave", () => setTimeout(dismiss, 1500));

  toastContainer.append(toast);

  // Keep at most 4 toasts visible
  while (toastContainer.children.length > 4) {
    toastContainer.firstElementChild.remove();
  }
};

const toastTypeForStatus = (status) => {
  if (status === 429) return "error";
  if (status === 413) return "warning";
  if (status === 400) return "warning";
  return "error";
};

const toastTitleForStatus = (status) => {
  if (status === 429) return "Rate Limited";
  if (status === 413 || status === 400) return "Input Blocked";
  return "Request Failed";
};

const PHONETIC_MAP = {
  "eye": "i", "aye": "i", "ay": "a", "ee": "e", "ess": "s",
  "arr": "r", "ar": "r", "are": "r", "aitch": "h", "ach": "h",
  "jay": "j", "kay": "k", "cue": "q", "que": "q", "pee": "p",
  "tee": "t", "dee": "d", "bee": "b", "cee": "c", "see": "c",
  "sea": "c", "gee": "g", "vee": "v", "wye": "y", "why": "y",
  "you": "u", "oh": "o", "em": "m", "en": "n", "el": "l",
  "ex": "x", "zee": "z", "zed": "z", "eff": "f", "ef": "f",
  "double-u": "w", "double u": "w", "dubya": "w",
};

const PHONETIC_RE = new RegExp(
  "\\b(" +
    Object.keys(PHONETIC_MAP)
      .sort((a, b) => b.length - a.length)
      .map((k) => k.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&"))
      .join("|") +
    ")\\b",
  "gi"
);

const normalisePhonetic = (text) => {
  let result = text.toLowerCase();
  // Replace phonetic letter names
  result = result.replace(PHONETIC_RE, (m) => PHONETIC_MAP[m.toLowerCase()] || m);
  // Collapse spaced-out single letters (e.g. "a p i" → "api")
  result = result.replace(/(?<![a-zA-Z])([a-zA-Z])(?:\s+([a-zA-Z]))+(?![a-zA-Z])/g, (m) =>
    m.replace(/\s+/g, "")
  );
  return result.replace(/\s+/g, " ").trim();
};

const validateInputLocally = (text) => {
  if (text.length > MAX_MESSAGE_CHARS) {
    showToast(
      `Message is too long (${text.length.toLocaleString()} / ${MAX_MESSAGE_CHARS.toLocaleString()} characters).`,
      "warning"
    );
    return false;
  }

  const lowered = text.toLowerCase();
  const phonetic = normalisePhonetic(text);

  if (
    CLIENT_BLOCKED_PHRASES.some((phrase) => lowered.includes(phrase)) ||
    CLIENT_BLOCKED_PHRASES.some((phrase) => phonetic.includes(phrase))
  ) {
    showToast(
      "Your message contains content that Shef cannot process. Ask about recipes, ingredients, substitutions, or cooking help.",
      "warning"
    );
    return false;
  }

  return true;
};

function loadState() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (stored && Array.isArray(stored.chats)) {
      const chats = stored.chats
        .filter((chat) => chat && typeof chat.id === "string")
        .map((chat) => ({
          id: chat.id,
          title: typeof chat.title === "string" && chat.title ? chat.title : "New Chat",
          createdAt: chat.createdAt || nowIso(),
          updatedAt: chat.updatedAt || chat.createdAt || nowIso(),
          messages: Array.isArray(chat.messages) ? chat.messages : [],
        }))
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
        .slice(0, MAX_CHATS);
      if (chats.length > 0) {
        return {
          activeChatId: chats.some((chat) => chat.id === stored.activeChatId)
            ? stored.activeChatId
            : chats[0].id,
          chats,
        };
      }
    }
  } catch (error) {
    console.warn("Unable to load saved chats.", error);
  }
  return { activeChatId: null, chats: [] };
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function createWelcomeMessage() {
  return {
    id: createId(),
    role: "assistant",
    text: WELCOME_TEXT,
    attachments: [],
    createdAt: nowIso(),
    isWelcome: true,
  };
}

function getSortedChats() {
  return [...state.chats].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
}

function getActiveChat() {
  return state.chats.find((chat) => chat.id === state.activeChatId) || null;
}

function deleteOldestChat() {
  const oldest = [...state.chats].sort((a, b) => new Date(a.updatedAt) - new Date(b.updatedAt))[0];
  if (!oldest) return;
  state.chats = state.chats.filter((chat) => chat.id !== oldest.id);
}

function createChat({ confirmLimit = false } = {}) {
  if (state.chats.length >= MAX_CHATS) {
    if (
      confirmLimit &&
      !window.confirm(
        "You already have 10 saved chats. Starting a new chat will delete your oldest conversation from this browser."
      )
    ) {
      return null;
    }
    deleteOldestChat();
  }

  const createdAt = nowIso();
  const chat = {
    id: createId(),
    title: "New Chat",
    createdAt,
    updatedAt: createdAt,
    messages: [createWelcomeMessage()],
  };
  state.chats.push(chat);
  state.activeChatId = chat.id;
  saveState();
  return chat;
}

function ensureActiveChat() {
  if (!getActiveChat()) {
    createChat();
  }
}

const setError = (message = "") => {
  composerError.textContent = message;
};

const setComposerBusy = (busy) => {
  isSending = busy;
  composerForm.classList.toggle("is-busy", busy);
  messageInput.disabled = busy;
  addButton.disabled = busy;
  micButton.disabled = busy;
  sendButton.disabled = busy;
};

const scrollToBottom = () => {
  requestAnimationFrame(() => {
    messageList.scrollTop = messageList.scrollHeight;
  });
};

const openSidebar = () => {
  sidebarBackdrop.hidden = false;
  document.body.classList.add("sidebar-open");
  sidebar.classList.add("is-open");
  sidebarBackdrop.classList.add("is-open");
  menuButton.setAttribute("aria-expanded", "true");
  sidebarCloseButton.focus();
};

const closeSidebar = () => {
  sidebar.classList.remove("is-open");
  sidebarBackdrop.classList.remove("is-open");
  document.body.classList.remove("sidebar-open");
  menuButton.setAttribute("aria-expanded", "false");

  window.setTimeout(() => {
    if (!sidebar.classList.contains("is-open")) {
      sidebarBackdrop.hidden = true;
    }
  }, 180);
};

const PRIVACY_COPY = {
  enabled: {
    intro:
      "Shef logs anonymous usage events only to understand whether this project is being used. Events may include sending a chat, selecting a recipe, uploading an image, recording audio, or seeing an error.",
    retention:
      "Shef does not store chat messages, uploaded images, audio recordings, generated recipes, personal identifiers, or exact IP addresses.",
    terms:
      "Use Shef for recipe and cooking help only. Do not submit personal, secret, unsafe, or non-recipe content.",
    usageAcceptance: "I understand Shef records anonymous usage events.",
    termsAcceptance: "I accept Shef's privacy and recipe-use terms.",
  },
  disabled: {
    intro: "Shef is not collecting usage analytics or anonymous usage events.",
    retention:
      "Recipe chats are saved only in this browser for your local chat history. Shef does not store chat messages, uploaded images, audio recordings, generated recipes, personal identifiers, exact IP addresses, or usage sessions.",
    terms:
      "Requests are processed only to answer your recipe question. Use Shef for recipe and cooking help only, and do not submit personal, secret, unsafe, or non-recipe content.",
    usageAcceptance: "I understand Shef is not collecting usage analytics.",
    termsAcceptance: "I accept Shef's privacy and recipe-use terms.",
  },
};

const applyPrivacyCopy = () => {
  const copy = USAGE_LOGGING_ENABLED ? PRIVACY_COPY.enabled : PRIVACY_COPY.disabled;
  if (privacyIntro) privacyIntro.textContent = copy.intro;
  if (privacyRetention) privacyRetention.textContent = copy.retention;
  if (privacyUseTerms) privacyUseTerms.textContent = copy.terms;
  if (privacyAcceptanceUsageText) {
    privacyAcceptanceUsageText.textContent = copy.usageAcceptance;
  }
  if (privacyAcceptanceTermsText) {
    privacyAcceptanceTermsText.textContent = copy.termsAcceptance;
  }
};

const hasAcceptedPrivacyTerms = () => {
  try {
    return Boolean(localStorage.getItem(PRIVACY_ACCEPTANCE_KEY));
  } catch (error) {
    console.warn("Unable to read privacy confirmation.", error);
    return false;
  }
};

const savePrivacyAcceptance = () => {
  try {
    localStorage.setItem(PRIVACY_ACCEPTANCE_KEY, nowIso());
  } catch (error) {
    console.warn("Unable to save privacy confirmation.", error);
  }
};

const updatePrivacyAcceptButton = () => {
  if (!privacyAcceptButton) return;
  privacyAcceptButton.disabled =
    privacyAcceptanceCheckboxes.length === 0 ||
    !privacyAcceptanceCheckboxes.every((checkbox) => checkbox.checked);
};

const setPrivacyModalRequired = (required) => {
  privacyConfirmationRequired = required;
  privacyModal?.classList.toggle("is-required", required);
  if (privacyCloseButton) {
    privacyCloseButton.hidden = required;
    privacyCloseButton.disabled = required;
  }
};

const openPrivacyModal = ({ requireConfirmation = false } = {}) => {
  if (!privacyModal) return;
  setPrivacyModalRequired(requireConfirmation);
  if (requireConfirmation) {
    privacyAcceptanceCheckboxes.forEach((checkbox) => {
      checkbox.checked = false;
    });
  }
  updatePrivacyAcceptButton();
  privacyModal.hidden = false;
  appShell?.setAttribute("inert", "");
  document.body.classList.add("modal-open");
  (requireConfirmation ? privacyAcceptanceCheckboxes[0] : privacyCloseButton)?.focus();
};

const closePrivacyModal = ({ restoreFocus = true } = {}) => {
  if (!privacyModal) return;
  if (privacyConfirmationRequired) return;
  privacyModal.hidden = true;
  appShell?.removeAttribute("inert");
  document.body.classList.remove("modal-open");
  if (restoreFocus) {
    privacyButton?.focus();
  }
};

const acceptPrivacyTerms = () => {
  if (!privacyAcceptButton || privacyAcceptButton.disabled) return;
  savePrivacyAcceptance();
  setPrivacyModalRequired(false);
  closePrivacyModal({ restoreFocus: false });
  messageInput?.focus();
};

const createAvatar = (kind) => {
  const avatar = document.createElement("div");
  avatar.className = kind === "user" ? "avatar user-avatar" : "avatar bot-avatar";
  avatar.setAttribute("aria-hidden", "true");

  if (kind === "user") {
    avatar.textContent = "U";
    return avatar;
  }

  avatar.innerHTML = '<img class="chef-logo-img" src="./assets/icon-transparent.png" alt="" />';
  return avatar;
};

const openLightbox = (src) => {
  const overlay = document.createElement("div");
  overlay.className = "lightbox-overlay";

  const closeBtn = document.createElement("button");
  closeBtn.className = "lightbox-close";
  closeBtn.type = "button";
  closeBtn.innerHTML = "&times;";
  closeBtn.setAttribute("aria-label", "Close lightbox");

  const img = document.createElement("img");
  img.className = "lightbox-image";
  img.src = src;
  img.alt = "Full size image";

  const dismiss = () => {
    overlay.classList.add("is-leaving");
    overlay.addEventListener("animationend", () => overlay.remove(), { once: true });
  };

  closeBtn.addEventListener("click", dismiss);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) dismiss();
  });

  const keyHandler = (e) => {
    if (e.key === "Escape") {
      dismiss();
      window.removeEventListener("keydown", keyHandler);
    }
  };
  window.addEventListener("keydown", keyHandler);

  overlay.append(closeBtn, img);
  document.body.append(overlay);
};

const parseDurationLabel = (label) => {
  if (!label) return 0;
  const parts = label.split(":").map(Number);
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return 0;
};

const createVoicePlayer = (attachment) => {
  const player = document.createElement("div");
  player.className = "voice-player";

  const playBtn = document.createElement("button");
  playBtn.className = "voice-player-btn";
  playBtn.type = "button";
  playBtn.setAttribute("aria-label", "Play voice message");

  const PLAY_ICON =
    '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="6,4 20,12 6,20"/></svg>';
  const PAUSE_ICON =
    '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="4" height="16" rx="1"/><rect x="15" y="4" width="4" height="16" rx="1"/></svg>';

  playBtn.innerHTML = PLAY_ICON;

  const waveform = document.createElement("div");
  waveform.className = "voice-waveform";
  const barCount = 28;
  for (let i = 0; i < barCount; i++) {
    const bar = document.createElement("span");
    bar.className = "waveform-bar";
    bar.style.setProperty("--bar-h", `${20 + Math.random() * 80}%`);
    bar.style.setProperty("--bar-i", i);
    waveform.append(bar);
  }

  const duration = document.createElement("span");
  duration.className = "voice-player-time";
  duration.textContent = attachment.durationLabel || "00:00";

  let audio = null;
  let isPlaying = false;
  let animFrame = null;
  const totalSeconds = parseDurationLabel(attachment.durationLabel);

  const updateProgress = () => {
    if (!audio || !isPlaying) return;
    const current = audio.currentTime;
    const total = audio.duration || totalSeconds;
    const progress = total > 0 ? current / total : 0;

    duration.textContent = formatDuration(Math.floor(current));

    const bars = waveform.querySelectorAll(".waveform-bar");
    bars.forEach((bar, i) => {
      bar.classList.toggle("is-played", i / bars.length <= progress);
    });

    animFrame = requestAnimationFrame(updateProgress);
  };

  const resetPlayer = () => {
    isPlaying = false;
    cancelAnimationFrame(animFrame);
    playBtn.innerHTML = PLAY_ICON;
    playBtn.setAttribute("aria-label", "Play voice message");
    player.classList.remove("is-playing");
    duration.textContent = attachment.durationLabel || "00:00";
    waveform
      .querySelectorAll(".waveform-bar")
      .forEach((bar) => bar.classList.remove("is-played"));
  };

  playBtn.addEventListener("click", () => {
    if (!audio) {
      audio = new Audio(attachment.dataUrl);
      audio.addEventListener("ended", resetPlayer);
    }

    if (isPlaying) {
      audio.pause();
      isPlaying = false;
      cancelAnimationFrame(animFrame);
      playBtn.innerHTML = PLAY_ICON;
      playBtn.setAttribute("aria-label", "Play voice message");
      player.classList.remove("is-playing");
    } else {
      audio.play();
      isPlaying = true;
      playBtn.innerHTML = PAUSE_ICON;
      playBtn.setAttribute("aria-label", "Pause voice message");
      player.classList.add("is-playing");
      updateProgress();
    }
  });

  player.append(playBtn, waveform, duration);
  return player;
};

const createStreamingDots = () => {
  const dots = document.createElement("span");
  dots.className = "streaming-dots";
  dots.setAttribute("aria-label", "Shef is typing");
  dots.innerHTML = `
    <span></span>
    <span></span>
    <span></span>
  `;
  return dots;
};

const isTouchInteraction = () =>
  window.matchMedia("(hover: none), (pointer: coarse), (max-width: 700px)").matches;

const createDescriptionLine = (label, value) => {
  const line = document.createElement("p");
  const strong = document.createElement("strong");
  strong.textContent = label;
  line.append(strong, document.createTextNode(value ? ` - ${value}` : " - "));
  return line;
};

const createRecipeOptionDescription = (option) => {
  const description = document.createElement("div");
  description.className = "recipe-option-description";
  description.append(
    createDescriptionLine("Flavor & Texture", option.descriptions?.flavorTexture || "Balanced, cozy, and rice-ready."),
    createDescriptionLine("Occasion & Fit", option.descriptions?.occasionFit || "A practical choice for everyday cooking."),
    createDescriptionLine("Pro Tip", option.descriptions?.proTip || "Prep aromatics before heating the pan.")
  );
  return description;
};

const closeRecipeOptionPreview = () => {
  recipeOptionPreview = null;
  const modal = document.querySelector(".recipe-option-modal");
  modal?.remove();
  document.body.classList.remove("modal-open");
};

const renderRecipeOptionPreview = () => {
  document.querySelector(".recipe-option-modal")?.remove();
  if (!recipeOptionPreview) return;

  const { messageId, option } = recipeOptionPreview;
  const overlay = document.createElement("div");
  overlay.className = "recipe-option-modal";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-labelledby", "recipeOptionPreviewTitle");

  const dialog = document.createElement("div");
  dialog.className = "recipe-option-dialog";

  const closeButton = document.createElement("button");
  closeButton.className = "recipe-option-close";
  closeButton.type = "button";
  closeButton.innerHTML = "&times;";
  closeButton.setAttribute("aria-label", "Close recipe description");

  const title = document.createElement("h2");
  title.id = "recipeOptionPreviewTitle";
  title.textContent = option.title;

  const description = createRecipeOptionDescription(option);

  const confirmButton = document.createElement("button");
  confirmButton.className = "recipe-option-confirm";
  confirmButton.type = "button";
  confirmButton.textContent = "Choose this Recipe";
  confirmButton.disabled = isSending;
  confirmButton.addEventListener("click", () => {
    if (isSending) return;
    closeRecipeOptionPreview();
    selectRecipeOption(messageId, option.title);
  });

  closeButton.addEventListener("click", closeRecipeOptionPreview);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      closeRecipeOptionPreview();
    }
  });

  dialog.append(closeButton, title, description, confirmButton);
  overlay.append(dialog);
  document.body.append(overlay);
  document.body.classList.add("modal-open");
  closeButton.focus();
};

const openRecipeOptionPreview = (messageId, option) => {
  recipeOptionPreview = { messageId, option };
  renderRecipeOptionPreview();
};

const createRecipeOptionsNode = ({ messageId, options, selectedRecipeTitle = "" }) => {
  const container = document.createElement("div");
  container.className = "recipe-options";

  const heading = document.createElement("p");
  heading.className = "recipe-options-heading";
  heading.textContent = "Choose one recipe:";
  container.append(heading);

  const list = document.createElement("div");
  list.className = "recipe-options-list";

  options.forEach((option) => {
    const isSelected = selectedRecipeTitle === option.title;
    const hasSelection = Boolean(selectedRecipeTitle);

    const item = document.createElement("div");
    item.className = "recipe-option-card";
    item.classList.toggle("is-selected", isSelected);
    item.classList.toggle("is-disabled", hasSelection || isSending);

    const titleButton = document.createElement("button");
    titleButton.className = "recipe-option-title";
    titleButton.type = "button";
    titleButton.textContent = option.title;
    titleButton.disabled = hasSelection || isSending;
    titleButton.setAttribute("aria-label", `Select ${option.title}`);

    titleButton.addEventListener("click", () => {
      if (hasSelection || isSending) return;
      if (isTouchInteraction()) {
        openRecipeOptionPreview(messageId, option);
        return;
      }
      selectRecipeOption(messageId, option.title);
    });

    const description = createRecipeOptionDescription(option);

    if (isSelected) {
      const badge = document.createElement("span");
      badge.className = "recipe-option-selected";
      badge.textContent = "Selected";
      item.append(titleButton, badge, description);
    } else {
      item.append(titleButton, description);
    }

    list.append(item);
  });

  container.append(list);
  return container;
};

const createMessageNode = ({
  id,
  role,
  text,
  attachments = [],
  createdAt,
  isStreaming = false,
  recipeOptions = [],
  selectedRecipeTitle = "",
}) => {
  const article = document.createElement("article");
  article.className = `message message-${role === "user" ? "user" : "assistant"}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role === "user" ? "user-bubble" : "assistant-bubble"}`;

  if (role === "assistant") {
    if (Array.isArray(recipeOptions) && recipeOptions.length >= 3 && !isStreaming) {
      bubble.append(
        createRecipeOptionsNode({
          messageId: id,
          options: recipeOptions,
          selectedRecipeTitle,
        })
      );
    } else if (text) {
      const content = renderAssistantText(text);
      if (isStreaming) {
        const cursor = document.createElement("span");
        cursor.className = "streaming-cursor";
        cursor.setAttribute("aria-hidden", "true");
        content.append(cursor);
      }
      bubble.append(content);
    } else if (isStreaming) {
      bubble.append(createStreamingDots());
    }
  } else if (text) {
    const paragraph = document.createElement("p");
    paragraph.innerHTML = escapeHtml(text).replaceAll("\n", "<br />");
    bubble.append(paragraph);
  }

  const hasImageOnly =
    !text &&
    attachments.length === 1 &&
    attachments[0].type === "image" &&
    attachments[0].dataUrl;

  if (hasImageOnly) {
    bubble.classList.add("bubble-image-only");
  }

  attachments.forEach((attachment) => {
    if (attachment.type === "image" && attachment.dataUrl) {
      const imgWrapper = document.createElement("div");
      imgWrapper.className = "bubble-image-wrapper";
      const img = document.createElement("img");
      img.className = "bubble-image";
      img.src = attachment.dataUrl;
      img.alt = attachment.name || "Attached image";
      img.addEventListener("click", () => openLightbox(attachment.dataUrl));
      imgWrapper.append(img);
      bubble.append(imgWrapper);
    } else if (attachment.type === "audio" && attachment.dataUrl) {
      bubble.append(createVoicePlayer(attachment));
    } else {
      const attachmentLine = document.createElement("p");
      attachmentLine.className = "message-attachment";
      attachmentLine.textContent =
        attachment.type === "audio"
          ? `Attached audio: ${attachment.name}`
          : `Attached image: ${attachment.name}`;
      bubble.append(attachmentLine);
    }
  });

  const footer = document.createElement("footer");
  footer.innerHTML =
    role === "user"
      ? `${formatClock(new Date(createdAt))} <span aria-hidden="true">sent</span>`
      : formatClock(new Date(createdAt));
  bubble.append(footer);

  if (role === "user") {
    article.append(bubble, createAvatar("user"));
  } else {
    article.append(createAvatar("assistant"), bubble);
  }

  return article;
};

function renderMessages() {
  const chat = getActiveChat();
  messageList.innerHTML = "";

  const divider = document.createElement("div");
  divider.className = "date-divider";
  divider.textContent = "Today";
  messageList.append(divider);

  if (!chat) return;

  chat.messages.forEach((message) => {
    messageList.append(createMessageNode(message));
  });
  scrollToBottom();
}

function renderChatList() {
  chatList.innerHTML = "";
  const chats = getSortedChats();

  if (!chats.length) {
    const empty = document.createElement("p");
    empty.className = "empty-chat-list";
    empty.textContent = "No saved chats yet.";
    chatList.append(empty);
    return;
  }

  chats.forEach((chat) => {
    const lastUserMessage = [...chat.messages].reverse().find((message) => message.role === "user");
    const row = document.createElement("button");
    row.className = `chat-row ${chat.id === state.activeChatId ? "is-active" : ""}`;
    row.type = "button";
    row.dataset.chatId = chat.id;
    row.innerHTML = `
      <span class="chat-row-icon" aria-hidden="true">S</span>
      <span class="chat-row-copy">
        <strong>${escapeHtml(chat.title)}</strong>
        <small>${escapeHtml(lastUserMessage?.text || "Ready for ingredients")}</small>
      </span>
      <time>${escapeHtml(formatChatTime(chat.updatedAt))}</time>
    `;
    chatList.append(row);
  });
}

function renderApp() {
  renderChatList();
  renderMessages();
}

function updateChatTitle(chat, message, attachments) {
  if (chat.title !== "New Chat") return;
  const attachmentLabel = attachments.length ? `${attachments[0].type} attachment` : "";
  const seed = (message || attachmentLabel || "Recipe Chat").trim();
  chat.title = seed.length > 34 ? `${seed.slice(0, 31)}...` : seed;
}

const showTyping = () => {
  hideTyping();

  typingNode = document.createElement("article");
  typingNode.className = "message message-assistant";
  typingNode.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">
      <img class="chef-logo-img" src="./assets/icon-transparent.png" alt="" />
    </div>
    <div class="bubble assistant-bubble typing-bubble" aria-label="Shef is typing">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  messageList.append(typingNode);
  scrollToBottom();
};

const hideTyping = () => {
  if (typingNode) {
    typingNode.remove();
    typingNode = null;
  }
};

const renderAttachments = () => {
  attachmentTray.innerHTML = "";

  if (imageAttachment) {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    chip.innerHTML = `
      <img src="${imageAttachment.previewUrl}" alt="" />
      <span>
        <strong>${escapeHtml(imageAttachment.file.name)}</strong>
        <small>${fileSize(imageAttachment.file.size)}</small>
      </span>
      <button class="remove-attachment" type="button" aria-label="Remove image attachment">x</button>
    `;
    chip.querySelector("button").addEventListener("click", () => {
      URL.revokeObjectURL(imageAttachment.previewUrl);
      imageAttachment = null;
      imageInput.value = "";
      renderAttachments();
    });
    attachmentTray.append(chip);
  }

  if (audioAttachment) {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    chip.innerHTML = `
      <span class="audio-icon" aria-hidden="true">WAV</span>
      <span>
        <strong>${escapeHtml(audioAttachment.name)}</strong>
        <small>${audioAttachment.durationLabel}</small>
      </span>
      <button class="remove-attachment" type="button" aria-label="Remove audio attachment">x</button>
    `;
    chip.querySelector("button").addEventListener("click", () => {
      audioAttachment = null;
      renderAttachments();
    });
    attachmentTray.append(chip);
  }
};

const clearComposer = () => {
  messageInput.value = "";
  if (imageAttachment) {
    URL.revokeObjectURL(imageAttachment.previewUrl);
  }
  imageAttachment = null;
  audioAttachment = null;
  imageInput.value = "";
  renderAttachments();
};

function historyForApi(chat) {
  return chat.messages
    .filter((message) => !message.isWelcome && (message.role === "user" || message.role === "assistant"))
    .slice(-12)
    .map((message) => ({ role: message.role, text: message.text }));
}

const parseSseEvent = (rawEvent) => {
  const event = { type: "message", data: null };
  const dataLines = [];

  rawEvent
    .replace(/\r\n/g, "\n")
    .split("\n")
    .forEach((line) => {
      if (line.startsWith("event:")) {
        event.type = line.slice(6).trim() || "message";
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    });

  if (dataLines.length) {
    event.data = JSON.parse(dataLines.join("\n"));
  }

  return event;
};

const processSseBuffer = (buffer, onEvent) => {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const events = normalized.split("\n\n");
  const remainder = events.pop() || "";

  events.forEach((rawEvent) => {
    if (rawEvent.trim()) {
      onEvent(parseSseEvent(rawEvent));
    }
  });

  return remainder;
};

const callChatApi = async ({
  message,
  imageFile,
  audioBlob,
  audioName,
  threadId,
  history,
  responseMode = RESPONSE_MODE_AUTO,
  usageEvent = "",
  onMeta,
  onChunk,
}) => {
  const formData = new FormData();
  formData.append("message", message);
  formData.append("thread_id", threadId);
  formData.append("history", JSON.stringify(history));
  formData.append("response_mode", responseMode);
  formData.append("usage_event", usageEvent);

  if (imageFile) {
    formData.append("image", imageFile, imageFile.name);
  }

  if (audioBlob) {
    formData.append("audio", audioBlob, audioName || "voice-note.wav");
  }

  let response;
  try {
    response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
      },
      body: formData,
    });
  } catch (networkErr) {
    const err = new Error("Could not reach Shef. Check your connection and try again.");
    err.status = 0;
    throw err;
  }

  if (!response.ok) {
    let payload = null;
    try {
      const text = await response.text();
      payload = text ? JSON.parse(text) : null;
    } catch {
      // Response body was not valid JSON, handled below.
    }
    const detail = payload?.detail || `Chat API returned ${response.status}`;
    const err = new Error(detail);
    err.status = response.status;
    throw err;
  }

  if (!response.body) {
    throw new Error("Chat API response did not include a readable stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamedText = "";
  let finalReply = "";
  let receivedDone = false;
  let effectiveResponseMode = responseMode;

  const handleEvent = ({ type, data }) => {
    if (type === "meta") {
      effectiveResponseMode = data?.response_mode || data?.responseMode || effectiveResponseMode;
      onMeta?.(effectiveResponseMode);
      return;
    }

    if (type === "delta" && typeof data?.text === "string") {
      streamedText += data.text;
      onChunk?.(data.text, effectiveResponseMode);
      return;
    }

    if (type === "done" && typeof data?.reply === "string") {
      finalReply = data.reply;
      effectiveResponseMode = data?.response_mode || data?.responseMode || effectiveResponseMode;
      receivedDone = true;
      return;
    }

    if (type === "error") {
      const err = new Error(data?.detail || "Shef could not respond. Check the server and try again.");
      err.status = data?.status || 500;
      throw err;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = processSseBuffer(buffer, handleEvent);
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    processSseBuffer(`${buffer}\n\n`, handleEvent);
  }

  return {
    reply: receivedDone ? finalReply : streamedText,
    responseMode: effectiveResponseMode,
  };
};

const sendMessage = async ({
  messageOverride = null,
  responseMode = RESPONSE_MODE_AUTO,
  sourceOptionMessageId = "",
  selectedRecipeTitle = "",
} = {}) => {
  if (isSending) return;

  const chat = getActiveChat();
  if (!chat) return;

  const hasMessageOverride = typeof messageOverride === "string";
  const message = (hasMessageOverride ? messageOverride : messageInput.value).trim();
  const imageFile = hasMessageOverride ? null : imageAttachment?.file || null;
  const audioBlob = hasMessageOverride ? null : audioAttachment?.blob || null;
  const audioName = hasMessageOverride
    ? "voice-note.wav"
    : audioAttachment?.name || "voice-note.wav";

  if (!message && !imageFile && !audioBlob) {
    showToast("Type a message, attach an image, or record audio first.", "warning");
    return;
  }

  if (message && !validateInputLocally(message)) {
    return;
  }

  setError("");
  const submittedAttachments = [];
  if (imageAttachment) {
    const imageDataUrl = await fileToDataUrl(imageAttachment.file);
    submittedAttachments.push({
      type: "image",
      name: imageAttachment.file.name,
      size: imageAttachment.file.size,
      dataUrl: imageDataUrl,
    });
  }
  if (audioAttachment) {
    const audioDataUrl = await fileToDataUrl(audioAttachment.blob);
    submittedAttachments.push({
      type: "audio",
      name: audioAttachment.name,
      durationLabel: audioAttachment.durationLabel,
      dataUrl: audioDataUrl,
    });
  }

  const apiHistory = historyForApi(chat);
  const chatId = chat.id;
  const previousTitle = chat.title;
  const optionMessage = sourceOptionMessageId
    ? chat.messages.find((item) => item.id === sourceOptionMessageId)
    : null;
  const previousSelectedRecipeTitle = optionMessage?.selectedRecipeTitle || "";
  const createdAt = nowIso();
  const userMessageId = createId();
  const assistantMessageId = createId();

  if (optionMessage && selectedRecipeTitle) {
    optionMessage.selectedRecipeTitle = selectedRecipeTitle;
  }

  chat.messages.push({
    id: userMessageId,
    role: "user",
    text: message,
    attachments: submittedAttachments,
    createdAt,
  });
  chat.updatedAt = createdAt;
  updateChatTitle(chat, message, submittedAttachments);
  saveState();
  chat.messages.push({
    id: assistantMessageId,
    role: "assistant",
    text: "",
    attachments: [],
    createdAt,
    isStreaming: true,
  });
  renderApp();
  if (!hasMessageOverride) {
    clearComposer();
  }
  setComposerBusy(true);

  let streamedReply = "";
  let effectiveResponseMode = responseMode;
  const updateStreamingReply = (text, isFinal = false) => {
    const targetChat = state.chats.find((item) => item.id === chatId);
    const assistantMessage = targetChat?.messages.find((item) => item.id === assistantMessageId);
    if (!assistantMessage) return;

    assistantMessage.text = text;
    assistantMessage.isStreaming = !isFinal;
    if (isFinal) {
      delete assistantMessage.isStreaming;
    }

    if (state.activeChatId === chatId) {
      renderMessages();
    }
  };

  try {
    const apiResult = await callChatApi({
      message,
      imageFile,
      audioBlob,
      audioName,
      threadId: chatId,
      history: apiHistory,
      responseMode,
      usageEvent: sourceOptionMessageId ? "recipe_selected" : "",
      onMeta: (mode) => {
        effectiveResponseMode = mode;
      },
      onChunk: (chunk, mode) => {
        effectiveResponseMode = mode;
        if (mode === RESPONSE_MODE_RECIPE_OPTIONS) {
          return;
        }
        streamedReply += chunk;
        updateStreamingReply(streamedReply);
      },
    });
    const reply = apiResult.reply;
    effectiveResponseMode = apiResult.responseMode;

    const targetChat = state.chats.find((item) => item.id === chatId);
    if (targetChat) {
      const assistantMessage = targetChat.messages.find((item) => item.id === assistantMessageId);
      const replyAt = nowIso();
      if (assistantMessage) {
        const parsedOptions =
          effectiveResponseMode === RESPONSE_MODE_RECIPE_OPTIONS ? parseRecipeOptions(reply) : [];
        assistantMessage.text = reply;
        assistantMessage.recipeOptions = parsedOptions.length >= 3 ? parsedOptions : [];
        assistantMessage.createdAt = replyAt;
        delete assistantMessage.isStreaming;
        targetChat.updatedAt = replyAt;
        saveState();
      }
    }
  } catch (error) {
    // Remove the failed user message from history
    const targetChat = state.chats.find((item) => item.id === chatId);
    if (targetChat) {
      targetChat.messages = targetChat.messages.filter(
        (m) => m.id !== userMessageId && m.id !== assistantMessageId
      );
      const targetOptionMessage = sourceOptionMessageId
        ? targetChat.messages.find((item) => item.id === sourceOptionMessageId)
        : null;
      if (targetOptionMessage) {
        targetOptionMessage.selectedRecipeTitle = previousSelectedRecipeTitle;
      }
      targetChat.title = previousTitle;
      targetChat.updatedAt = targetChat.messages.length
        ? targetChat.messages[targetChat.messages.length - 1].createdAt
        : targetChat.createdAt;
      saveState();
    }

    // Restore the typed text so the user doesn't lose their input
    if (!hasMessageOverride) {
      messageInput.value = message;
    }

    const toastType = toastTypeForStatus(error.status);
    showToast(
      error.message || "Shef could not respond. Check the server and try again.",
      toastType,
      5000,
      toastTitleForStatus(error.status)
    );
  } finally {
    setComposerBusy(false);
    renderApp();
  }
};

function selectRecipeOption(messageId, recipeTitle) {
  sendMessage({
    messageOverride: `Show me the recipe for ${recipeTitle}.`,
    responseMode: RESPONSE_MODE_FULL_RECIPE,
    sourceOptionMessageId: messageId,
    selectedRecipeTitle: recipeTitle,
  });
}

const startRecordingTimer = () => {
  recordingStartedAt = Date.now();
  recordingTimer.textContent = "00:00";
  clearInterval(recordingInterval);
  recordingInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - recordingStartedAt) / 1000);
    recordingTimer.textContent = formatDuration(elapsed);
  }, 250);
};

const stopRecordingTimer = () => {
  clearInterval(recordingInterval);
  recordingInterval = null;
};

function mergeAudioChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const samples = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    samples.set(chunk, offset);
    offset += chunk.length;
  });
  return samples;
}

function resampleAudio(samples, inputSampleRate, outputSampleRate) {
  if (!inputSampleRate || inputSampleRate === outputSampleRate) {
    return samples;
  }

  const ratio = inputSampleRate / outputSampleRate;
  const length = Math.max(1, Math.round(samples.length / ratio));
  const resampled = new Float32Array(length);

  for (let index = 0; index < length; index += 1) {
    const sourceIndex = index * ratio;
    const before = Math.floor(sourceIndex);
    const after = Math.min(before + 1, samples.length - 1);
    const weight = sourceIndex - before;
    resampled[index] = samples[before] * (1 - weight) + samples[after] * weight;
  }

  return resampled;
}

function writeString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeWav(samples, sampleRate) {
  const wavSamples = resampleAudio(samples, sampleRate, TARGET_AUDIO_SAMPLE_RATE);
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + wavSamples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + wavSamples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, TARGET_AUDIO_SAMPLE_RATE, true);
  view.setUint32(28, TARGET_AUDIO_SAMPLE_RATE * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, wavSamples.length * bytesPerSample, true);

  let offset = 44;
  for (let index = 0; index < wavSamples.length; index += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, wavSamples[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

const startRecording = async () => {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!navigator.mediaDevices?.getUserMedia || !AudioContextClass) {
    setError("Microphone recording is not supported in this browser.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const context = new AudioContextClass();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const silentGain = context.createGain();
    const chunks = [];

    silentGain.gain.value = 0;
    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };

    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(context.destination);

    recorder = {
      stream,
      context,
      source,
      processor,
      silentGain,
      chunks,
      sampleRate: context.sampleRate,
    };

    setError("");
    micButton.classList.add("is-recording");
    recordingBar.hidden = false;
    startRecordingTimer();
  } catch (error) {
    setError("Microphone permission was blocked or unavailable.");
  }
};

const stopRecording = async () => {
  if (!recorder) return;

  const activeRecorder = recorder;
  recorder = null;
  activeRecorder.processor.disconnect();
  activeRecorder.source.disconnect();
  activeRecorder.silentGain.disconnect();
  activeRecorder.stream.getTracks().forEach((track) => track.stop());
  await activeRecorder.context.close();

  const durationSeconds = Math.max(1, Math.round((Date.now() - recordingStartedAt) / 1000));
  const samples = mergeAudioChunks(activeRecorder.chunks);
  stopRecordingTimer();
  recordingBar.hidden = true;
  micButton.classList.remove("is-recording");

  if (!samples.length) {
    setError("No audio was captured. Try recording again.");
    return;
  }

  audioAttachment = {
    blob: encodeWav(samples, activeRecorder.sampleRate),
    name: "voice-note.wav",
    durationLabel: formatDuration(durationSeconds),
  };
  renderAttachments();
};

composerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

addButton.addEventListener("click", () => {
  imageInput.click();
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    setError("Choose an image file.");
    imageInput.value = "";
    return;
  }

  if (imageAttachment) {
    URL.revokeObjectURL(imageAttachment.previewUrl);
  }

  imageAttachment = {
    file,
    previewUrl: URL.createObjectURL(file),
  };
  setError("");
  renderAttachments();
});

micButton.addEventListener("click", () => {
  if (recorder) {
    stopRecording();
    return;
  }

  startRecording();
});

stopRecordingButton.addEventListener("click", stopRecording);

menuButton.addEventListener("click", openSidebar);
sidebarCloseButton.addEventListener("click", closeSidebar);
sidebarBackdrop.addEventListener("click", closeSidebar);
privacyButton?.addEventListener("click", openPrivacyModal);
privacyCloseButton?.addEventListener("click", closePrivacyModal);
privacyAcceptanceCheckboxes.forEach((checkbox) => {
  checkbox.addEventListener("change", updatePrivacyAcceptButton);
});
privacyAcceptButton?.addEventListener("click", acceptPrivacyTerms);
privacyModal?.addEventListener("click", (event) => {
  if (event.target === privacyModal) {
    closePrivacyModal();
  }
});

chatList.addEventListener("click", (event) => {
  const row = event.target.closest(".chat-row");
  if (!row) return;
  state.activeChatId = row.dataset.chatId;
  saveState();
  clearComposer();
  setError("");
  renderApp();
  closeSidebar();
});

window.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }
  if (recipeOptionPreview) {
    closeRecipeOptionPreview();
    return;
  }
  if (privacyModal && !privacyModal.hidden) {
    closePrivacyModal();
    return;
  }
  if (sidebar.classList.contains("is-open")) {
    closeSidebar();
    menuButton.focus();
  }
});

window.matchMedia("(min-width: 901px)").addEventListener("change", (event) => {
  if (event.matches) {
    closeSidebar();
  }
});

newChatButton.addEventListener("click", () => {
  const chat = createChat({ confirmLimit: true });
  if (!chat) return;
  clearComposer();
  setError("");
  renderApp();
  closeSidebar();
});

ensureActiveChat();
saveState();
renderApp();
applyPrivacyCopy();
if (!hasAcceptedPrivacyTerms()) {
  openPrivacyModal({ requireConfirmation: true });
}
