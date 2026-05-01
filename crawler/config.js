require('dotenv').config();

const getArg = function (flag) {
    const i = args.indexOf(flag);
    if (i == -1) {
        return [false, null]
    }

    return [true, args[i + 1]]; // a bit buggy but works for now (assumes flags are followed by their value, and no flags are repeated)
};

const getParsedArgs = function () {
    args = process.argv.slice(2);

    // Screenshots are always enabled; --screenshots-dir overrides the output directory.
    const [, screenshotsDirValue] = getArg('--screenshots-dir');

    // Progress directory:
    // --progress-dir overrides ENV/CONFIG when provided.
    const [, progressDirValue] = getArg('--progress-dir');

    return {
        urlsArg: getArg('--urls')[1],          // JSON string of URLs (direct)
        urlsFile: getArg('--urls-file')[1] ,   // File path with URLs (for large datasets)
        outputFile: getArg('--output')[1] || `./results/ads_${Date.now()}.json`, // TODO: move this to ENV config file.
        screenshotBaseDir: screenshotsDirValue || process.env.SCREENSHOT_DIR || CONFIG.SCREENSHOT_DIR,
        // Progress files target directory for writeProgress
        progressDir: progressDirValue || process.env.PROGRESS_DIR || CONFIG.PROGRESS_DIR,
        workerId: getArg('--worker-id')[1] ,
        forceAuth: getArg('--force-auth')[0],  // Force authenticated crawl mode
        setupProfile: getArg('--setup-profile')[0], // Run one-time profile setup
        advertiserID: args[0] || 'AR16735076323512287233',
        max: parseInt(args[1]) || 10,
    }
};

const CONFIG = {
    // Display (adjust for your environment)
    XVFB_SWITCH: parseInt(process.env.XVFB_SWITCH) || 0,                    // 0 = local display, 1 = Xvfb (headless servers)
    HEADLESS: process.env.HEADLESS !== 'false',                   // true = no GUI (faster), false = show browser
    VIEWPORT_WIDTH: parseInt(process.env.VIEWPORT_WIDTH) || 1480,              // Browser viewport width
    VIEWPORT_HEIGHT: parseInt(process.env.VIEWPORT_HEIGHT) || 1480,             // Browser viewport height
    BROWSER_EXECUTABLE_PATH: process.env.BROWSER_EXECUTABLE_PATH || '/usr/bin/google-chrome',
    USER_AGENT: process.env.USER_AGENT || 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',

    // Authentication
    PROFILE_DIR: process.env.PROFILE_DIR || './browser_profile',  // Where to save/load login session

    // Performance
    PAGE_LOAD_WAIT: parseInt(process.env.PAGE_LOAD_WAIT) || 4000,              // Wait after page loads (ms, min 3000)
    PAGE_TIMEOUT: parseInt(process.env.PAGE_TIMEOUT) || 60000,               // Max time per page (ms)
    PROTOCOL_TIMEOUT: parseInt(process.env.PROTOCOL_TIMEOUT) || 120000,        // Max time for CDP protocol calls like screenshot (ms)
    BROWSER_RESTART_INTERVAL: parseInt(process.env.BROWSER_RESTART_INTERVAL) || 50,      // Restart browser every N ads (prevent memory leaks)

    // Error Handling
    MAX_RETRIES: parseInt(process.env.MAX_RETRIES) || 3,                    // Retries per ad on failure
    RETRY_DELAY: parseInt(process.env.RETRY_DELAY) || 2000,                 // Wait between retries (ms)
    MAX_CONSECUTIVE_AUTH_FAILURES: parseInt(process.env.MAX_CONSECUTIVE_AUTH_FAILURES) || 5,  // Max consecutive auth failures before aborting

    // Progress & Features
    SAVE_INTERVAL: parseInt(process.env.SAVE_INTERVAL) || 3,                  // Save progress every N ads
    DOWNLOAD_VARIATIONS: process.env.DOWNLOAD_VARIATIONS !== 'false',         // Download carousel ad variations
    SCROLL_DELAY: parseInt(process.env.SCROLL_DELAY) || 3000,                // Delay between scrolls (advertiser mode only, ms)
    PROGRESS_DIR: process.env.PROGRESS_DIR || './results/progress',              // Where to save progress files

    SCREENSHOT_DIR: process.env.SCREENSHOT_DIR || './results/screenshots',        // Where to save screenshots (if enabled)
    BASE_URL: 'https://adstransparency.google.com',

    MAX_SCROLL_ATTEMPTS: parseInt(process.env.MAX_SCROLL_ATTEMPTS) || 10,          // Max scroll attempts to load creatives (advertiser mode)
};

const BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--no-first-run',
    '--no-zygote',
    '--disable-gpu',
    '--disable-blink-features=AutomationControlled',
    '--disable-features=IsolateOrigins,site-per-process'
];

module.exports = {
    CONFIG,
    BROWSER_ARGS,
    getParsedArgs,
};