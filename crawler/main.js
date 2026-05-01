const fs = require('fs');
const { CONFIG, BROWSER_ARGS, getParsedArgs } = require('./config');
const { saveADs, loadUrlArg, loadUrlFile } = require('./io');
const { extractCreatives } = require('./creative');
const puppeteer = require('puppeteer');

const args = getParsedArgs();

function isProtocolOrTimeoutError(err) {
    if (!err) return false;
    const msg = err.message || String(err);
    return err.name === 'ProtocolError' || msg.includes('Network.enable') || msg.includes('timed out') || msg.includes('ProtocolError');
}

process.on('uncaughtException', (err) => {
    if (isProtocolOrTimeoutError(err)) {
        console.error(`⚠️  [uncaughtException] ProtocolError caught — Chrome overloaded. Progress saved. Exiting cleanly.\n   ${err.message}`);
        process.exit(0);
    }
    console.error(`❌ [uncaughtException] Fatal error: ${err.message}`);
    process.exit(1);
});

process.on('unhandledRejection', (reason) => {
    if (isProtocolOrTimeoutError(reason)) {
        // This fires from puppeteer's internal event listeners (e.g. Network.enable timed out).
        // The crawling loop is still alive — the next browser.newPage() will fail, triggering
        // the existing browser-restart logic. Just log and continue.
        console.error(`⚠️  [unhandledRejection] ProtocolError from puppeteer internals — browser restart will be triggered.\n   ${reason?.message || reason}`);
        return;
    }
    console.error(`❌ [unhandledRejection] Fatal: ${reason?.message || reason}`);
    process.exit(1);
});

async function launchBrowser() {
    // Launch browser (with or without profile)
    const allowProfileCreation = args.setupProfile;
    const forceAuth = args.forceAuth || args.setupProfile;
    const headless = args.setupProfile ? false : CONFIG.HEADLESS;
    console.log(`🚀 Launching browser!\n\theadless: ${headless}\n\tprofile: ${CONFIG.PROFILE_DIR}\n\tAuth mode: ${forceAuth ? "Profile" : "None"}`);

    if (forceAuth) {
        console.log(`🔐 Using authenticated profile: ${CONFIG.PROFILE_DIR}`);
    }

    const browserArgs = [...BROWSER_ARGS];
    if (CONFIG.XVFB_SWITCH === 1) {
        browserArgs.push(`--display=${xvfb._display}`);
    }

    const launchOptions = {
        executablePath: CONFIG.BROWSER_EXECUTABLE_PATH,
        headless,
        args: browserArgs,
        protocolTimeout: CONFIG.PROTOCOL_TIMEOUT
    };

    // Add profile if requested (normal auth or setup-profile)
    if (forceAuth) {
        // Check profile exists (unless we're creating it for first time via --setup-profile)
        if (!allowProfileCreation && !fs.existsSync(CONFIG.PROFILE_DIR)) {
            throw new Error(`❌ Auth profile not found at ${CONFIG.PROFILE_DIR}. Run with --setup-profile first.`);
        }
        launchOptions.userDataDir = CONFIG.PROFILE_DIR;
    }

    return await puppeteer.launch(launchOptions);
}

async function scrapeURLs(urls) {
    console.log(`📖 Processing ${urls.length} URLs... (Worker ${args.workerId ?? "Main"})`);
    let browser = await launchBrowser();

    // this helps us to have access to fresh browser instance even if the browser needs to be restarted periodically (e.g. to prevent memory leaks)
    const browserLauncher = async (forceRestart = false) => {
        if (browser.connected && forceRestart) {
            await browser.close()
            browser = await launchBrowser();
            return browser;
        }

        if (!browser.connected) {
            console.log(`⚠️ Browser was disconnected. Relaunching...`);
            browser = await launchBrowser();
            return browser;
        }

        return browser;
    }

    try {
        return await extractCreatives(urls, browserLauncher);
    } catch (error) {
        // Let's see what are our errors.
        throw error;
    } finally {
        await browser.close();
    }
}

async function collectUrls(page) {
    const creativeUrls = [];

    let scrolls = 0, noNew = 0;
    while (creativeUrls.length < args.max && scrolls < CONFIG.MAX_SCROLL_ATTEMPTS && noNew < 3) {
        const prevCount = creativeUrls.length;

        const found = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('creative-preview')).map(p => {
                const a = p.querySelector('a[href*="/creative/"]');
                const img = p.querySelector('img[src*="archive/simgad"]');
                if (!a) return null;
                const href = a.getAttribute('href');
                const crMatch = href.match(/\/creative\/(CR[\d]+)/);
                const arMatch = href.match(/\/advertiser\/(AR[\d]+)/);
                return crMatch && arMatch ? {
                    creativeID: crMatch[1], advertiserID: arMatch[1],
                    relativeURL: href, imageUrl: img?.src || null
                } : null;
            }).filter(Boolean);
        });

        const existing = new Set(creativeUrls.map(u => u.creativeID));
        for (const f of found) {
            if (!existing.has(f.creativeID) && creativeUrls.length < args.max) {
                creativeUrls.push(f);
                existing.add(f.creativeID);
            }
        }

        console.log(`  ${creativeUrls.length} unique creatives...`);
        noNew = creativeUrls.length === prevCount ? noNew + 1 : 0;

        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        await new Promise(r => setTimeout(r, CONFIG.SCROLL_DELAY)); // TODO: why this is needed?
        scrolls++;
    }

    return creativeUrls.slice(0, args.max);
}

async function scrapeGoogleAdTransparency(advertiserUrl) {
    const browser = await launchBrowser();
    const page = await browser.newPage();


    // TODO: duplicated logic with creative.js - refactor to reuse
    await page.setViewport({ width: CONFIG.VIEWPORT_WIDTH, height: CONFIG.VIEWPORT_HEIGHT });
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36');

    console.log(`Navigating to: ${advertiserUrl}`);
    await page.goto(advertiserUrl, { waitUntil: 'networkidle0', timeout: 30000 });

    try {
        await page.waitForSelector('creative-preview', { timeout: 15000 });
    } catch (e) {
        console.log('creative-preview not found, continuing...');
    }


    console.log('Step 1: Scrolling to load creatives...');
    creativeUrls = await collectUrls(page);
    browser.close();

    console.log(`\nStep 2: Visiting ${creativeUrls.length} creative pages...`);
    return await scrapeURLs(creativeUrls)
}

async function runEngine() {
    // Read URLs from file or argument, or default to advertiser mode
    if (args.urlsFile) {
        console.log(`Reading URLs from file: ${args.urlsFile}`);
        const urls = loadUrlFile(args.urlsFile);
        return await scrapeURLs(urls);
    }

    // URL list mode from argument (backwards compatibility)
    if (args.urlsArg) {
        console.log(`Reading URLs from argument`);
        const urls = loadUrlArg(args.urlsArg);
        return await scrapeURLs(urls);
    }

    // Default to advertiser mode if no URLs provided
    console.log(`\n📋 Advertiser Mode: ${args.advertiserID} (max: ${args.max})`);
    return await scrapeGoogleAdTransparency(
        `https://adstransparency.google.com/advertiser/${args.advertiserID}?region=US`
    );
}

// ============ AUTHENTICATION SETUP ============

/**
 * Setup authentication profile - opens browser for manual login
 */
async function setupAuthProfile() {
    console.log('🔐 Opening browser for authentication...');
    console.log(`   Profile will be saved to: ${CONFIG.PROFILE_DIR}`);
    console.log('\n📝 Steps:');
    console.log('   1. Login to your Google account');
    console.log('   2. Navigate to a few ad pages to verify login works');
    console.log('   3. Close the browser when done\n');

    // Launch browser with profile (always GUI for login, allow creating new profile)
    const browser = await launchBrowser(); // You need to pass the args in the ENV config file

    // Use the existing page that opens with the profile
    const pages = await browser.pages();
    const page = pages[0] || await browser.newPage();
    await page.setViewport({ width: CONFIG.VIEWPORT_WIDTH, height: CONFIG.VIEWPORT_HEIGHT });

    // Navigate to Google Ads Transparency Center
    await page.goto('https://www.google.com', { waitUntil: 'networkidle0' });

    console.log('✅ Browser opened - Login and close when done');
    console.log('   Waiting for browser to close...');

    // Wait for browser to close
    await new Promise((resolve) => {
        browser.on('disconnected', resolve);
    });

    console.log('\n✅ Profile saved successfully!');
    console.log(`   Location: ${CONFIG.PROFILE_DIR}`);
    console.log('   Future crawls will use this profile for region-restricted ads');
}

async function main() {
    // Handle setup-profile mode (no Xvfb needed - runs in GUI)
    if (args.setupProfile) { // TODO: this needs to be tested!
        return await setupAuthProfile();
    }

    const ads = await runEngine();

    saveADs(ads, args.outputFile);
}

main()

module.exports = {
    main,
};
