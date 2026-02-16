const API_BASE = `${window.location.origin}/api`;

let isReady = false;

// Rotating placeholder phrases
const placeholders = [
    'imagine a meme where everything is fine...',
    'that one where the guy looks back...',
    'something about mondays...',
    'the dog in the burning room...',
    'when you pretend to understand...',
    'that vibe when the plan falls apart...',
    'a cat judging you silently...',
    'the audacity of hope but make it funny...',
    'two people shaking hands over something weird...',
    'me explaining something nobody asked about...',
];
let placeholderIndex = 0;
let placeholderTimer = null;
let placeholderActive = false;

// DOM elements
const searchInput = document.getElementById('searchInput');
const searchPlaceholder = document.getElementById('searchPlaceholder');
const searchButton = document.getElementById('searchButton');
const suggestionsContainer = document.getElementById('suggestions');
const resultsContainer = document.getElementById('resultsContainer');
const memeGrid = document.getElementById('memeGrid');
const emptyState = document.getElementById('emptyState');
const resultsCount = document.getElementById('resultsCount');

// Animated placeholder fade-up effect
function showNextPlaceholder() {
    if (!placeholderActive) return;

    const current = placeholders[placeholderIndex];
    searchPlaceholder.textContent = current;

    // Fade in
    searchPlaceholder.classList.remove('fade-out');
    searchPlaceholder.classList.add('fade-in');

    // After visible for 2.5s, fade out then show next
    placeholderTimer = setTimeout(() => {
        if (!placeholderActive) return;
        searchPlaceholder.classList.remove('fade-in');
        searchPlaceholder.classList.add('fade-out');

        placeholderTimer = setTimeout(() => {
            placeholderIndex = (placeholderIndex + 1) % placeholders.length;
            showNextPlaceholder();
        }, 300);
    }, 2500);
}

function stopPlaceholderAnimation() {
    placeholderActive = false;
    if (placeholderTimer) {
        clearTimeout(placeholderTimer);
        placeholderTimer = null;
    }
    searchPlaceholder.classList.remove('fade-in', 'fade-out');
    searchPlaceholder.style.opacity = '0';
}

function startPlaceholderAnimation() {
    stopPlaceholderAnimation();
    placeholderActive = true;
    showNextPlaceholder();
}

// Initialize
async function init() {
    await checkStatus();
    setupEventListeners();
    startPlaceholderAnimation();

    // Poll status until ready
    if (!isReady) {
        const interval = setInterval(async () => {
            await checkStatus();
            if (isReady) {
                clearInterval(interval);
            }
        }, 2000);
    }
}

// Check backend status
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        isReady = data.status === 'ready';

        if (isReady) {
            searchInput.disabled = false;
            searchButton.disabled = false;
        } else {
            searchInput.disabled = true;
            searchButton.disabled = true;
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Search on button click
    searchButton.addEventListener('click', handleSearch);

    // Search on enter key
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Stop animation on focus, restart on blur if empty
    searchInput.addEventListener('focus', () => {
        stopPlaceholderAnimation();
    });

    searchInput.addEventListener('blur', () => {
        if (!searchInput.value.trim()) {
            startPlaceholderAnimation();
        }
    });

    // Hide placeholder when user types, show when cleared
    searchInput.addEventListener('input', () => {
        if (searchInput.value.trim()) {
            stopPlaceholderAnimation();
        }
    });

    // Suggestion chips
    const suggestionChips = document.querySelectorAll('.suggestion-chip');
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            stopPlaceholderAnimation();
            searchInput.value = chip.textContent;
            handleSearch();
        });
    });
}

// Handle search
async function handleSearch() {
    const query = searchInput.value.trim();

    if (!query || !isReady) return;

    // Hide empty state
    emptyState.classList.add('hidden');

    // Show loading in results
    resultsContainer.classList.add('visible');
    memeGrid.innerHTML = '<div class="loading">searching</div>';

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                limit: 20
            })
        });

        const data = await response.json();

        // Update results count
        resultsCount.textContent = `${data.count} result${data.count !== 1 ? 's' : ''} for "${query}"`;

        // Clear loading and render results
        memeGrid.innerHTML = '';
        data.results.forEach((meme, index) => {
            const card = createMemeCard(meme, index);
            memeGrid.appendChild(card);
        });

        if (data.count === 0) {
            memeGrid.innerHTML = '<div class="loading">no results found. try a different description.</div>';
        }

    } catch (error) {
        console.error('Search error:', error);
        memeGrid.innerHTML = '<div class="loading">error performing search. check console.</div>';
    }
}

// Create meme card element
function createMemeCard(meme, index) {
    const card = document.createElement('div');
    card.className = 'meme-card';
    card.style.animationDelay = `${index * 0.05}s`;

    // Format score as percentage
    const scorePercent = Math.round(meme.score * 100);

    card.innerHTML = `
        <div class="meme-image-container">
            <img
                class="meme-image"
                src="${meme.url}"
                alt="${meme.name}"
                loading="lazy"
            >
        </div>
        <div class="meme-info">
            <h3 class="meme-name">${meme.name}</h3>
            <div class="meme-meta">
                <span class="meme-source">${meme.source}</span>
                <div class="meme-score">
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${scorePercent}%"></div>
                    </div>
                    <span>${scorePercent}%</span>
                </div>
            </div>
        </div>
    `;

    // Click to open meme creation modal
    card.addEventListener('click', () => {
        openMemeModal(meme);
    });

    return card;
}

// Meme Modal Functions
function openMemeModal(meme) {
    const modal = document.getElementById('memeModal');
    const modalImage = document.getElementById('modalMemeImage');
    const descriptionInput = document.getElementById('memeDescription');
    const generatedMemeContainer = document.getElementById('generatedMemeContainer');
    const memeActionsMini = document.getElementById('memeActionsMini');

    // Set modal content
    modalImage.src = meme.url;
    descriptionInput.value = '';
    generatedMemeContainer.style.display = 'none';
    memeActionsMini.classList.remove('visible');

    // Store meme data for generation
    modal.dataset.memeId = meme.id;
    modal.dataset.memeName = meme.name;
    modal.dataset.memeUrl = meme.url;

    // Show modal
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeMemeModal() {
    const modal = document.getElementById('memeModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

async function generateMeme() {
    const modal = document.getElementById('memeModal');
    const descriptionInput = document.getElementById('memeDescription');
    const generateButton = document.getElementById('generateMemeButton');
    const generatedMemeContainer = document.getElementById('generatedMemeContainer');
    const memeActionsMini = document.getElementById('memeActionsMini');
    const loadingIndicator = document.getElementById('generationLoading');

    const description = descriptionInput.value.trim();

    if (!description) {
        alert('Please describe what you want the meme to say!');
        return;
    }

    // Show loading
    loadingIndicator.style.display = 'block';
    generateButton.disabled = true;
    memeActionsMini.classList.remove('visible');

    try {
        const response = await fetch(`${API_BASE}/generate-meme`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                template_id: modal.dataset.memeId,
                template_name: modal.dataset.memeName,
                template_url: modal.dataset.memeUrl,
                description: description
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to generate meme');
        }

        // Resolve to a full URL (handles both absolute and relative paths)
        const memeUrl = data.meme_url.startsWith('http')
            ? data.meme_url
            : `${window.location.origin}${data.meme_url}`;

        // Replace template image with generated meme
        const modalImage = document.getElementById('modalMemeImage');
        modalImage.src = memeUrl;

        // Show mini action buttons below the meme
        memeActionsMini.classList.add('visible');

        // Store URL for copying/downloading
        generatedMemeContainer.dataset.memeUrl = memeUrl;
        generatedMemeContainer.style.display = 'block';

    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}\n\nMake sure you've added your ANTHROPIC_API_KEY to the .env file.`);
    } finally {
        loadingIndicator.style.display = 'none';
        generateButton.disabled = false;
    }
}

function copyMemeLink() {
    const generatedMemeContainer = document.getElementById('generatedMemeContainer');
    const memeUrl = generatedMemeContainer.dataset.memeUrl;

    if (memeUrl) {
        navigator.clipboard.writeText(memeUrl).then(() => {
            const copyButton = document.getElementById('copyLinkButton');
            const originalText = copyButton.textContent;
            copyButton.textContent = '✓ Copied!';
            setTimeout(() => {
                copyButton.textContent = originalText;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy:', err);
            alert('Failed to copy link');
        });
    }
}

function downloadMeme() {
    const generatedMemeContainer = document.getElementById('generatedMemeContainer');
    const memeUrl = generatedMemeContainer.dataset.memeUrl;

    if (memeUrl) {
        const link = document.createElement('a');
        link.href = memeUrl;
        link.download = 'meme.jpg';
        link.click();
    }
}

// Initialize on page load
init();
