/**
 * ShipAI Frontend — Connects to live backend API
 */

const API_BASE = 'http://localhost:8000';

// ── Nav scroll effect ──
window.addEventListener('scroll', () => {
  const nav = document.getElementById('mainNav');
  if (window.scrollY > 50) {
    nav.classList.add('scrolled');
  } else {
    nav.classList.remove('scrolled');
  }
});

// ── Animate stats on scroll ──
function animateCounter(el, target, duration = 1500) {
  let start = 0;
  const step = target / (duration / 16);
  function update() {
    start += step;
    if (start >= target) {
      el.textContent = target;
      return;
    }
    el.textContent = Math.floor(start);
    requestAnimationFrame(update);
  }
  update();
}

const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(document.getElementById('stat-endpoints'), 18);
      animateCounter(document.getElementById('stat-patterns'), 16);
      animateCounter(document.getElementById('stat-infra'), 7);
      animateCounter(document.getElementById('stat-tests'), 51);
      statsObserver.disconnect();
    }
  });
}, { threshold: 0.5 });

const statsBar = document.querySelector('.stats-bar');
if (statsBar) statsObserver.observe(statsBar);

// ── Intersection Observer for fade-in ──
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.glass-card').forEach(card => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(30px)';
  card.style.transition = 'all 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
  fadeObserver.observe(card);
});

// ── Hardware Check ──
async function checkHardware() {
  const btn = document.getElementById('hwCheckBtn');
  const results = document.getElementById('hwResults');
  
  btn.textContent = 'Detecting...';
  btn.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/api/system/hardware`);
    const data = await resp.json();

    document.getElementById('hw-os').textContent = data.system.os;
    document.getElementById('hw-cpu').textContent = data.cpu.name;
    document.getElementById('hw-ram').textContent = `${data.memory.total_gb} GB (${data.memory.available_gb} GB free)`;
    document.getElementById('hw-gpu').textContent = data.gpu ? data.gpu.name : 'None detected';
    document.getElementById('hw-vram').textContent = data.gpu ? `${data.gpu.vram_total_mb} MB (${data.gpu.vram_free_mb} MB free)` : 'N/A';
    document.getElementById('hw-disk').textContent = `${data.disk.free_gb} GB`;
    document.getElementById('hw-tier-label').textContent = data.recommendation.tier_label;
    document.getElementById('hw-models').textContent = `Recommended: ${data.recommendation.recommended_models.join(', ')}`;

    results.classList.add('visible');
    btn.textContent = 'Hardware Detected!';
  } catch (err) {
    btn.textContent = 'Could not connect to ShipAI backend';
    console.error(err);
  }

  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = 'Re-detect Hardware';
  }, 2000);
}

// ── Pricing (load from API) ──
async function loadPricing() {
  const grid = document.getElementById('pricingGrid');
  
  try {
    const resp = await fetch(`${API_BASE}/api/license/tiers`);
    const data = await resp.json();
    
    grid.innerHTML = data.tiers.map((tier, i) => `
      <div class="glass-card pricing-card ${i === 2 ? 'featured' : ''}">
        <h3>${tier.label}</h3>
        <div class="pricing-amount">${tier.price.replace('/month', '')}<span>${tier.price.includes('/') ? '/mo' : ''}</span></div>
        <ul class="pricing-features">
          ${tier.features.map(f => `<li>${f}</li>`).join('')}
        </ul>
        <button class="pricing-btn ${i === 2 ? 'primary' : ''}">${i === 0 ? 'Start Free' : i === 3 ? 'Contact Sales' : 'Get Started'}</button>
      </div>
    `).join('');
  } catch (err) {
    // Fallback if API not running
    grid.innerHTML = `
      <div class="glass-card pricing-card">
        <h3>Free</h3>
        <div class="pricing-amount">$0</div>
        <ul class="pricing-features">
          <li>2 project templates</li>
          <li>3 production infra patterns</li>
          <li>Up to 3 projects</li>
          <li>Community support</li>
        </ul>
        <button class="pricing-btn">Start Free</button>
      </div>
      <div class="glass-card pricing-card">
        <h3>Starter</h3>
        <div class="pricing-amount">$19<span>/mo</span></div>
        <ul class="pricing-features">
          <li>All 3 project templates</li>
          <li>5 production infra patterns</li>
          <li>Up to 10 projects</li>
          <li>Email support</li>
        </ul>
        <button class="pricing-btn">Get Started</button>
      </div>
      <div class="glass-card pricing-card featured">
        <h3>Pro</h3>
        <div class="pricing-amount">$49<span>/mo</span></div>
        <ul class="pricing-features">
          <li>All templates + custom</li>
          <li>All 7 infra patterns</li>
          <li>Unlimited projects</li>
          <li>Builder Bot (no-code)</li>
          <li>Priority support</li>
        </ul>
        <button class="pricing-btn primary">Get Started</button>
      </div>
      <div class="glass-card pricing-card">
        <h3>Enterprise</h3>
        <div class="pricing-amount">Custom</div>
        <ul class="pricing-features">
          <li>Everything in Pro</li>
          <li>White-label branding</li>
          <li>Custom scaling policies</li>
          <li>Dedicated support</li>
          <li>SLA guarantee</li>
        </ul>
        <button class="pricing-btn">Contact Sales</button>
      </div>
    `;
  }
}

loadPricing();

// ── Builder Bot Chat ──
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const messages = document.getElementById('chatMessages');
  const text = input.value.trim();
  if (!text) return;

  // Add user message
  messages.innerHTML += `<div class="msg user">${escapeHtml(text)}</div>`;
  input.value = '';
  messages.scrollTop = messages.scrollHeight;

  // Show typing indicator
  const typingId = 'typing-' + Date.now();
  messages.innerHTML += `<div class="msg bot" id="${typingId}" style="opacity:0.5;">Thinking...</div>`;
  messages.scrollTop = messages.scrollHeight;

  try {
    // First try advisor for architecture recommendations
    const advisorResp = await fetch(`${API_BASE}/api/advisor/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: text })
    });
    const advisorData = await advisorResp.json();

    // Remove typing indicator
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    let botResponse = '';

    // Show quick matches
    if (advisorData.quick_matches && advisorData.quick_matches.length > 0) {
      const matches = advisorData.quick_matches.slice(0, 3);
      botResponse += `<strong>Recommended Patterns:</strong><br>`;
      matches.forEach(m => {
        botResponse += `&#8226; <strong>${m.pattern}</strong> (${m.category}) - ${m.description}<br>`;
      });
      botResponse += `<br>`;
    }

    // Show LLM analysis
    if (advisorData.analysis) {
      botResponse += advisorData.analysis.replace(/\n/g, '<br>');
    }

    if (!botResponse) {
      botResponse = "I can help you build that! Try being more specific about what kind of AI product you need.";
    }

    messages.innerHTML += `<div class="msg bot">${botResponse}</div>`;
  } catch (err) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
    messages.innerHTML += `<div class="msg bot">I'm having trouble connecting to the ShipAI backend. Make sure it's running on port 8000.</div>`;
  }

  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
