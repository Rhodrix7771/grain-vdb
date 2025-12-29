const STRIPE_PUBLISHABLE_KEY = 'pk_live_51SjmeCJxncHJMxeFjsEaYjRXt0aNo4O0W8FMyaEqVpW2h30xfLXxtJIXC0B9c2AJXggKXW2mJ4oNB1aIa3E22QT200Mxhli8ug';

const STRIPE_LINKS = {
    starter: 'https://buy.stripe.com/fZu5kEgcm9HmcDle6p8AE00',
    pro: 'https://buy.stripe.com/4gM28s2lw9Hm6eX9Q98AE01',
    enterprise: 'mailto:hello@grainvdb.io?subject=Enterprise%20Inquiry'
};

const WALLETS = {
    eth: '0x9196E767E90B01A949826Cd26F1fC289901de3',
    btc: '14ZPxMQhQjNx6PkRkKmABwW9aNjXL4SMbp'
};

/**
 * Handles the checkout flow.
 */
async function initiateCheckout(tier, method = 'stripe') {
    if (method === 'stripe' && !STRIPE_LINKS[tier]) {
        console.warn(`[GrainVDB] Missing link for ${tier}.`);
        alert("Selection Pending: Please contact hello@grainvdb.io.");
        return;
    }

    if (method === 'crypto') {
        showCryptoModal(tier);
        return;
    }

    window.location.href = STRIPE_LINKS[tier];
}

function showCryptoModal(tier) {
    const prices = {
        starter: { eth: '0.05 ETH', btc: '0.0015 BTC' },
        pro: { eth: '0.2 ETH', btc: '0.038 BTC' }
    };

    const modalHtml = `
        <div id="crypto-modal" style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.95);display:flex;align-items:center;justify-content:center;z-index:9999;font-family:sans-serif; backdrop-filter: blur(10px);">
            <div style="background:#111;padding:40px;border-radius:20px;border:1px solid #50fa7b;max-width:440px;text-align:center;">
                <h2 style="color:#50fa7b;margin-bottom:10px;font-size:24px;">GrainVDB Alpha: ${tier.toUpperCase()}</h2>
                
                <div style="display:flex; gap:10px; margin-bottom:20px;">
                    <button id="btn-eth" onclick="selectCrypto('eth', '${tier}')" style="flex:1; background:#222; color:#fff; border:1px solid #333; padding:10px; border-radius:8px; cursor:pointer;">ETH/Base</button>
                    <button id="btn-btc" onclick="selectCrypto('btc', '${tier}')" style="flex:1; background:#222; color:#fff; border:1px solid #333; padding:10px; border-radius:8px; cursor:pointer;">BTC</button>
                </div>

                <div id="crypto-details">
                    <div style="background:#000;padding:20px;border-radius:12px;margin-bottom:20px;border:1px solid #333;">
                        <p id="crypto-amount" style="color:#50fa7b;font-size:24px;font-weight:bold;margin:0;">${prices[tier].eth}</p>
                        <p id="crypto-network" style="color:#555;font-size:11px;margin-top:4px;">(Network: Ethereum or Base L2)</p>
                    </div>

                    <div style="background:#000;padding:20px;border-radius:12px;margin-bottom:30px;border:1px solid #333;text-align:left;">
                        <code id="crypto-address" style="word-break:break-all;color:#fff;font-size:12px;">${WALLETS.eth}</code>
                    </div>
                </div>

                <button onclick="window.location.href='success.html'" style="background:#50fa7b;color:#000;border:none;padding:15px 40px;border-radius:50px;font-weight:bold;cursor:pointer;width:100%;">I Have Sent the Payment</button>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    window.selectCrypto = (type, tier) => {
        document.getElementById('crypto-amount').innerText = prices[tier][type];
        document.getElementById('crypto-address').innerText = WALLETS[type];
        document.getElementById('crypto-network').innerText = type === 'eth' ? '(Network: Ethereum or Base L2)' : '(Network: Bitcoin Mainnet)';
    };

    window.selectCrypto('eth', tier);
}

window.initiateCheckout = initiateCheckout;
