const { CONFIG, getParsedArgs } = require('./config');
const { createScreenshotDir, getScreenshotDir, writeProgress } = require('./io');
const { extractCreativeData } = require('./dataMiner');
const puppeteer = require('puppeteer');
const { TimeoutError, ProtocolError } = puppeteer;
const path = require('path');

function isTargetClosedError(e) {
    return e && (e.name === 'TargetCloseError' || (e.message && e.message.includes('Target closed')));
}

const args = getParsedArgs();

const NOT_FOUND_TRIGGERS = ["can't find ad", "kan advertentie niet vinden"];
const REGION_PHRASES = ['not found in your', 'in uw regio'];

// Prefer CLI-provided screenshot directory (via --screenshots) when available,
// otherwise fall back to CONFIG.SCREENSHOT_DIR.
const screenshotBaseDir = args.screenshotBaseDir || CONFIG.SCREENSHOT_DIR;
const screenshotDir = getScreenshotDir(screenshotBaseDir, args.workerId);

async function handleError(page, info) {
    if (!page || page.isClosed()) return null;
    let errorScreenshotPath = null;
    if (screenshotDir) {
        try {
            errorScreenshotPath = path.join(screenshotDir, `${info.creativeID}_error.png`);
            await page.screenshot({ path: errorScreenshotPath, fullPage: false });
            console.log(`   📸 Error screenshot saved`);
        } catch (e) {
            if (!isTargetClosedError(e)) console.log(`   ⚠️ Could not capture error screenshot`);
        }
    }
    try {
        if (!page.isClosed()) await page.close();
    } catch (e) {
        if (!isTargetClosedError(e)) console.log(`   ⚠️ Could not close page after error`);
    }
    return errorScreenshotPath;
}

// Haven't touched this.
async function checkCreativePageError(page) {
    return page.evaluate((notFoundTriggers, regionPhrases) => {
        const contentEl = document.querySelector('main, [role="main"]') || document.body;
        const bodyText = contentEl.innerText || '';
        const bodyLower = bodyText.toLowerCase();
        const isError = notFoundTriggers.some(t => bodyLower.includes(t));
        if (isError) {
            const lines = bodyText.split('\n').map(l => l.trim()).filter(l => l.length > 0 && l.length < 200);
            const errorMsg = lines.find(l => regionPhrases.some(p => l.toLowerCase().includes(p))) || 'Ad not found';
            const isRegionError = regionPhrases.some(p => errorMsg.toLowerCase().includes(p));
            const advertiserName = (
                document.querySelector('.advertiser-scope .advertiser-name')?.textContent?.trim() ||
                document.querySelector('.advertiser-scope [buttoncontent]')?.textContent?.trim() ||
                null
            );
            return { isError: true, message: errorMsg, isRegionError, advertiserName };
        }
        return { isError: false, message: null, isRegionError: false, advertiserName: null };
    }, NOT_FOUND_TRIGGERS, REGION_PHRASES);
}

async function getBrowserPage(browser) {
    const pages = await browser.pages();
    if (args.forceAuth && pages.length > 0) {
        const p = pages[0];
        if (!p.isClosed()) return p;
        // Page was closed (e.g. after timeout); open a new one so we don't use a dead target
    }
    return await browser.newPage();
}

// TODO: This function is doing a lot. We should break it down into smaller functions.
async function takeScreenshots(page, info) {
    try {
        // TODO: make this a function
        const containerEl = await page.$('.creative-container');
        if (containerEl) {
            await containerEl.evaluate(node => {
                node.style.display = 'block';
            });
        }

        // TODO: make this a function
        const itemElements = await page.$$('.creative-container .creative-sub-container');
        for (const el of itemElements) {
            await el.evaluate(node => {
                node.style.display = 'block';
            });
        }
        // TODO: get rid of this ugly syntax. Maybe a function?
        await new Promise(resolve => setTimeout(resolve, 1000)); // TODO: Adjust to ensure we don't get empty sshots

        const creativeSelector = '.has-variation > div'
        const elements = await page.$$(creativeSelector);

        if (elements.length === 0) {
            console.log(`   ⚠️ Selector ${creativeSelector} found no elements`);
        }

        const screenshotPaths = [];
        for (var idx = 0; idx < elements.length; idx++) {
            const element = elements[idx];
            const boundingBox = await element.boundingBox();
            const screenshotPath = path.join(screenshotDir, `${info.creativeID}-v${idx}.png`);

            if (!element || !boundingBox) {
                throw new Error(`Element ${creativeSelector} has no bounding box (possibly not visible)`);
            }

            await element.screenshot({ path: screenshotPath });
            console.log(`   📸 Screenshot for ad ${info.creativeID} variant ${idx} has been taken`);

            screenshotPaths.push(screenshotPath);
            if (!CONFIG.DOWNLOAD_VARIATIONS) {
                break;  // Only screenshot the first variation if not downloading all variations
            }
        }

        return screenshotPaths;
    } catch (e) {
        // Strategy 2: Fallback to full viewport
        console.log(`   ⚠️ Strategy 1 error: ${e.message} — falling back to full page`);
        const screenshotPath = path.join(screenshotDir, `${info.creativeID}_fullPage.png`);
        await page.screenshot({ path: screenshotPath, fullPage: false });
        console.log(`   📸 Screenshot (viewport fallback)`);
        return [screenshotPath];
    }

}

async function getAdsFromURL(info, getBrowser) {
    const url = info.relativeURL?.startsWith('http') ? info.relativeURL : `${CONFIG.BASE_URL}${info.relativeURL}`;
    let retries = 0;
    let success = false;
    let authFailures = 0;

    while (retries < CONFIG.MAX_RETRIES && !success) {
        const browser = await getBrowser();

        let page;
        try {
            page = await getBrowserPage(browser);
            page.setDefaultTimeout(CONFIG.PAGE_TIMEOUT);
        } catch (e) {
            // browser.newPage() can throw ProtocolError (e.g. Network.enable timed out)
            // if Chrome is overloaded — force a restart and retry
            retries++;
            console.error(`   ❌ Error (${retries}/${CONFIG.MAX_RETRIES}) [NEW PAGE]: ${e.message}`);
            await getBrowser(true).catch(() => {});
            if (retries < CONFIG.MAX_RETRIES) continue;
            return [{
                creativeID: info.creativeID,
                advertiserID: info.advertiserID,
                creativeURL: url,
                status: 'error',
                error: e.message,
                errorType: 'protocol',
                retries,
                timestamp: new Date().toISOString()
            }, authFailures];
        }

        try {
            console.log(`Attempting to scrape ${url} (Retry ${retries})`);
            console.log(`\n\t${info.creativeID}${retries > 0 ? ` (retry ${retries})` : ''}`);
            console.log(`\t${url}`);

            // Set viewport and user agent with error handling
            await page.setViewport({ width: CONFIG.VIEWPORT_WIDTH, height: CONFIG.VIEWPORT_HEIGHT });
            await page.setUserAgent(CONFIG.USER_AGENT);
            await page.goto(url, { waitUntil: 'networkidle0', timeout: CONFIG.PAGE_TIMEOUT });

            // Check for not-found / region-restricted BEFORE waiting for .creative-container.
            // These error pages never render the container, so waitForSelector would always
            // time out first and mask the real failure reason.
            const errorCheck = await checkCreativePageError(page);
            if (errorCheck.isError) {
                const errorScreenshotPath = await handleError(page, info);
                const errorType = errorCheck.isRegionError ? 'region_restricted' : 'not_found';
                console.log(`   ⚠️ ${errorType}: ${errorCheck.message}`);
                return [{
                    creativeID: info.creativeID,
                    advertiserID: info.advertiserID,
                    creativeURL: url,
                    advertiserName: errorCheck.advertiserName || info.advertiserName || null,
                    status: 'not_found',
                    error: errorCheck.message,
                    errorType,
                    isRegionError: errorCheck.isRegionError,
                    errorScreenshot: errorScreenshotPath || null,
                    timestamp: new Date().toISOString()
                }, authFailures];
            }

            await page.waitForSelector('.creative-container', { timeout: CONFIG.PAGE_TIMEOUT });

            const data = await extractCreativeData(page, info);

            data.screenshotPath = await takeScreenshots(page, info);

            await page.close();

            console.log(`   ✅ ${data.adFormat} | Image: ${data.imageUrl ? 'Yes' : 'No'} | Variations: ${data.variations?.length || 0} | Target: ${data.targetUrl?.slice(0, 50) || 'none'}`);
            return [data, 0];
        } catch (error) {
            if (!(error instanceof TimeoutError || error instanceof ProtocolError || error.message.includes('timeout') || error.message.includes('timed out'))) {
                throw error; // Only retry on timeout/protocol errors
            }

            retries++;
            console.error(`   ❌ Error (${retries}/${CONFIG.MAX_RETRIES}) [TIMEOUT]: ${error.message}`);

            if (retries < CONFIG.MAX_RETRIES) {
                // Browser may be in a broken state after a timeout — restart it before retrying
                console.error(`   🔄 Restarting browser before retry...`);
                await getBrowser(true).catch(() => {});
                await new Promise(r => setTimeout(r, CONFIG.RETRY_DELAY));
                continue; // Retry the same URL
            }

            // Take screenshot of error state and close page; guard against Target closed
            let errorScreenshotPath = null;
            if (screenshotDir && page && !page.isClosed()) {
                try {
                    errorScreenshotPath = path.join(screenshotDir, `${info.creativeID}_error.png`);
                    await page.screenshot({ path: errorScreenshotPath, fullPage: false });
                    console.log(`   📸 Error screenshot saved`);
                } catch (e) {
                    if (!isTargetClosedError(e)) console.log(`   ⚠️ Could not capture error screenshot: ${e.message}`);
                }
                try {
                    await page.close();
                } catch (e) {
                    if (!isTargetClosedError(e)) console.log(`   ⚠️ Could not close page: ${e.message}`);
                }
            }

            return [{
                creativeID: info.creativeID,
                advertiserID: info.advertiserID,
                creativeURL: url,
                status: 'error',
                error: error.message,
                errorType: 'timeout',
                retries: retries,
                errorScreenshot: errorScreenshotPath || undefined,
                timestamp: new Date().toISOString()
            }, authFailures];
        }
    }
}

async function extractCreatives(urls = [], getBrowser) {
    if (urls.length == 0) {
        console.log('⚠️ No creative URLs provided');
        return [];
    }

    createScreenshotDir(screenshotDir);

    const results = [];
    let consecutiveAuthFailures = 0;  // Track consecutive auth failures

    for (let i = 0; i < urls.length; i++) {
        const info = urls[i];
        console.log(`\n🔍 [${i + 1}/${urls.length}] Processing creative ID: ${info.creativeID}`);


        const [item, authFailures] = await getAdsFromURL(info, getBrowser);
        consecutiveAuthFailures += authFailures;

        if (authFailures < 1) {
            consecutiveAuthFailures = 0; // reset on success
        }

        results.push(item);

        // Progressive save
        if ((i + 1) % CONFIG.SAVE_INTERVAL === 0) {
            const progressDir = args.progressDir || CONFIG.PROGRESS_DIR;
            writeProgress(results, progressDir, args.workerId);
        }
    }

    return results;
}

module.exports = {
    extractCreatives
};