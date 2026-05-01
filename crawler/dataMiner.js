const { TimeoutError } = require('puppeteer');
const { CONFIG } = require('./config');

const CREATIVE_IMAGE_SELECTORS = [
    'html-renderer .html-container img[src*="archive/simgad"]',
    'html-renderer .html-container img[src*="tpc.googlesyndication.com"]',
    'html-renderer .html-container img[src^="http"]',
];

const GOOGLE_PATTERNS = [
    'google.com', 'google.co', 'google.net', 'google.org',
    'google/', 'safety.google', 'googlesyndication', 'googleusercontent',
    'youtube', 'ytimg', 'doubleclick', 'gstatic', 'googleadservices'
];

const APPLE_APP_PATTERNS = ['apps.apple.com', 'itunes.apple.com'];
const GOOGLE_APP_PATTERNS = ['play.google.com', 'market.android.com'];

async function extractImageUrls(creativePage) {
    return await creativePage.evaluate((selectors) => {
        // Target the canonical ad render container.
        // Google Ad Transparency always places the actual creative image inside
        // html-renderer > .html-container, which is specific to the rendered ad —
        // unlike the creative grid thumbnails that also use tpc.googlesyndication
        // but appear elsewhere in the DOM.
        for (const sel of selectors) {
            const img = document.querySelector(sel);
            if (img?.src && img.src.startsWith('http')) return img.src;
        }

        // Last resort: largest non-icon image
        const allImages = Array.from(document.querySelectorAll('img[src^="http"]'));
        const largeImages = allImages.filter(img =>
            img.naturalWidth > 100 &&
            img.naturalHeight > 100 &&
            !img.src.includes('icon') &&
            !img.src.includes('logo')
        );

        return largeImages[0]?.src || null;
    }, CREATIVE_IMAGE_SELECTORS);
}

function isAdLink(url) {
    return url && (url.includes('tpc.googlesyndication.com') || url.includes('simgad'));
}

/** Runs in browser context: collect img/iframe/a href|src URLs from current document. */
function collectLinkUrlsInDocument() {
    const links = [];
    ['img', 'iframe', 'a'].forEach(tag => {
        const attr = tag === 'a' ? 'href' : 'src';
        document.querySelectorAll(`${tag}[${attr}]`).forEach(el => links.push(el[attr]));
    });
    return links.filter(Boolean);
}

async function getIframAnchors(frame) {
    return await frame.evaluate((patterns) =>
        Array.from(document.querySelectorAll('a[href]'))
            // .filter(a => a.href?.startsWith('http') &&
            //     !patterns.some(p => a.href.toLowerCase().includes(p)))
            .map(a => ({
                href: a.href,
                id: a.id,
                text: a.textContent?.trim().replace(/\s+/g, ' ').slice(0, 100),
                innerText: a.innerText?.trim().replace(/\s+/g, ' ').slice(0, 100),
                adUrl: (() => {
                    try {
                        const v = new URL(a.href).searchParams.get('adurl');
                        return v ? v.trim() : a.href;
                    } catch { return a.href; }
                })(),
            }))
        , GOOGLE_PATTERNS);
}

async function getIframTextData(frame) {
    return await frame.evaluate(() => Array.from(document.querySelectorAll('h1, h2, h3, h4, p, button'))
        .map(el => el.textContent.trim())
        .filter(t => t.length > 1 && t.length < 500 && !t.startsWith('data:'))
    );
}

async function getIframVideoThumbnail(frame) {
    return await frame.evaluate(() => {
        const bgUrl = (el) => {
            if (!el) return null;
            const bg = el.style?.backgroundImage ||
                window.getComputedStyle(el).backgroundImage;
            const m = bg?.match(/url\(["']?([^"')]+)["']?\)/);
            return m?.[1] || null;
        };

        // 1. Thumbnail overlay image (most common for video ads)
        const startImg = document.querySelector(
            '.start-image, [class*="start-image"], [class*="thumbnail-image"]'
        );
        if (bgUrl(startImg)) return bgUrl(startImg);

        // 2. <video poster> — reliable when present
        const video = document.querySelector('video[poster]');
        if (video?.poster) return video.poster;

        // 3. <img> inside the video container
        const containerImg = document.querySelector(
            '#all-video-container img[src], .video-overlay img[src], .video-root-1 img[src]'
        );
        if (containerImg?.src?.startsWith('http')) return containerImg.src;

        // 4. Any element with background-image inside the video container
        const container = document.querySelector(
            '#all-video-container, .video-overlay, .video-root-1'
        );
        if (container) {
            for (const el of container.querySelectorAll('*')) {
                const url = bgUrl(el);
                if (url?.startsWith('http')) return url;
            }
        }

        return null;
    })
}

async function waitToLoadFrame(iframeHandle) {
    try {
        const frame = await iframeHandle.contentFrame();

        // Give it a short moment to populate
        await frame.waitForFunction(() => {
            return document.body && document.body.innerText.trim().length > 0;
        }, { timeout: 3000 });

        return frame;
    } catch (err) {
        if (err instanceof TimeoutError || err.message.includes('Waiting')) {
            return null; // skip if we cannot even get the body
        }
        throw err;
    }
}

/**
 * Returns the frame for variation 0 (landing page): the first .creative-sub-container's
 * ad iframe in DOM order. Used so targetUrl/adUrl/isClickable are scoped to the landing page only.
 */
async function getVariationZeroFrame(creativePage) {
    const selector = '.creative-container .creative-sub-container fletch-renderer iframe, ' +
        '.creative-container .creative-sub-container iframe[src*="safeframe"], ' +
        '.creative-container .creative-sub-container iframe[src*="adframe"]';
    const iframeHandle = await creativePage.$(selector);
    if (!iframeHandle) return null;
    return await waitToLoadFrame(iframeHandle);
}

async function extractIframData(creativePage, variation0Frame = null) {
    const results = {
        anchors: [],
        landingPageAnchors: [],
        iframeTexts: [],
        tpcLinks: [],
        frameLinks: [],
        videoThumbnailUrl: null,
    };

    for (const frame of creativePage.frames()) {
        if (frame.detached) {
            continue;
        }

        // TODO: check if we can get rid of the the timeout by adding some urls here.
        const frameUrl = frame.url();
        const hasAdContent = frameUrl && (frameUrl.includes('discover_ads') ||
            frameUrl.includes('pagead') || frameUrl.includes('simgad') ||
            frameUrl.includes('tpc.googlesyndication.com') || frameUrl.includes('archive/') ||
            frameUrl.includes('/adframe'));

        // Collect frame URL itself if it's an ad link
        if (isAdLink(frameUrl)) {
            results.tpcLinks.push(frameUrl);
        }

        if (!hasAdContent) {
            continue;
        }

        try {
            // Check if the frame is accessible. Otherwise frame.evaluate below
            // gets stuck for some frames, slowing down the crawler.
            // test case: AR11507905187872243713/creative/CR07497361827488071681
            await frame.waitForSelector('body', { timeout: 500 });
        } catch (err) {
            if (err instanceof TimeoutError || err.message.includes('Waiting')) {
                continue; // skip if we cannot even get the body
            }

            throw err;
        }

        const frameAnchors = await getIframAnchors(frame);
        results.anchors.push(...frameAnchors);
        // Single-value metadata (targetUrl, adUrl, isClickable) is from landing page (variation 0) only
        if (variation0Frame && frame === variation0Frame) {
            results.landingPageAnchors.push(...frameAnchors);
        }

        results.iframeTexts.push(...await getIframTextData(frame));
        results.frameLinks.push(...await frame.evaluate(collectLinkUrlsInDocument));

        if (!results.videoThumbnailUrl) {
            results.videoThumbnailUrl = await getIframVideoThumbnail(frame);
        }
    }

    return results;
}

async function extractPageData(creativeInfo, creativePage) {
    return await creativePage.evaluate((info) => {
        // --- Advertiser name ---
        let advertiserName = info.advertiserName;
        if (!advertiserName) {
            // Same order as error page: .advertiser-scope first, then page-level h1/.advertiser-name/[buttoncontent]
            advertiserName = document.querySelector('.advertiser-scope .advertiser-name')?.textContent?.trim() ||
                document.querySelector('.advertiser-scope [buttoncontent]')?.textContent?.trim() ||
                document.querySelector('h1')?.textContent?.trim() ||
                document.querySelector('.advertiser-name')?.textContent?.trim() ||
                document.querySelector('[buttoncontent]')?.textContent?.trim() ||
                null;
        }

        // --- Ad format: read directly from the metadata label Google provides ---
        // Google shows e.g. <div class="property"><strong>Format:</strong> Video</div>
        // For localized crawls the label and value may differ:
        //   EN: "Format:"  → values: "Video" | "Image" | "Text"
        //   NL: "Indeling:" → values: "Video" | "Afbeelding" | "Tekst"
        // Do NOT infer from iframe src — all ad iframes use generic URLs like /adframe or safeframe
        const FORMAT_LABELS = new Set(['format:', 'indeling:']);
        const FORMAT_VALUE_MAP = { 'afbeelding': 'image', 'tekst': 'text', 'video': 'video', 'image': 'image', 'text': 'text' };
        let adFormat = 'text';
        const formatProps = document.querySelectorAll('.properties .metadata .property');
        for (const prop of formatProps) {
            const label = prop.querySelector('strong');
            if (label && FORMAT_LABELS.has(label.textContent.trim().toLowerCase())) {
                const raw = prop.textContent.replace(label.textContent, '').trim().toLowerCase();
                adFormat = FORMAT_VALUE_MAP[raw] || raw || 'text';
                break;
            }
        }
        // Fallback: metadata div absent — check for a creative-scoped iframe (not page-level iframes)
        if (adFormat === 'text') {
            const adIframe = document.querySelector(
                '.creative-sub-container:not(.hidden) fletch-renderer iframe, ' +
                '.creative-sub-container:not(.hidden) creative iframe'
            );
            if (adIframe) adFormat = 'image';
        }

        // --- Variations: extract creativeId from fletch-renderer script URLs ---
        // fletch-renderer injects a <script src="...?creativeId=XXX&..."> for each variation;
        // that creativeId is the stable identifier for the variation asset
        const variations = [];
        const containers = document.querySelectorAll('.creative-sub-container');
        if (containers.length > 1) {
            containers.forEach((c, i) => {
                const script = c.querySelector('fletch-renderer script[src*="creativeId"]');
                if (script?.src) {
                    const match = script.src.match(/[?&]creativeId=(\d+)/);
                    variations.push({ index: i + 1, type: adFormat, url: script.src, creativeId: match?.[1] || null });
                    return;
                }
                // Legacy fallbacks for older/non-fletch ad formats
                const img = c.querySelector('img#marketing-image') || c.querySelector('img[src*="simgad"]');
                if (img?.src) { variations.push({ index: i + 1, type: 'image', url: img.src }); return; }
                const legacyIframe = c.querySelector('iframe[src*="archive/sadbundle"]');
                if (legacyIframe?.src) { variations.push({ index: i + 1, type: 'text', url: legacyIframe.src }); return; }
                variations.push({ index: i + 1, type: 'unknown', url: null });
            });
        }

        // --- iframeInfo: scope to ad-specific iframes, not all iframes on the page ---
        // fletch-renderer iframes and safeframe iframes are the actual ad containers
        const creativeEl = document.querySelector('creative') || document.body;
        const adIframes = creativeEl.querySelectorAll(
            'fletch-renderer iframe, iframe[src*="safeframe"], iframe[src*="adframe"]'
        );

        // --- Text ad content: extract <p> and heading text from the active creative ---
        // Scoped to div[ng-app] creative / .creative-sub-container-si to exclude page chrome
        // (nav, metadata, footer). Only meaningful for text-format ads but collected for all
        // so the field is always present and can be used for downstream filtering.
        // const adText = [];
        // const textScope = document.querySelector(
        //     'div[ng-app] .creative-sub-container-si, ' +
        //     'div[ng-app] .creative-sub-container:not(.hidden), ' +
        //     'div[ng-app] creative'
        // );
        // if (textScope) {
        //     adText = textScope.innerText.trim().split('\n').map(s => s.trim()).filter(s => s.length > 0);
        //     // const texts = Array.from(textScope.querySelectorAll('h1, h2, h3, h4, p'))
        //     //     .map(el => el.textContent.trim())
        //     //     .filter(t => t.length > 0);
        //     // if (texts.length > 0) adText = texts;
        // }
        // const itemElements = document.getElementsByClassName('creative-sub-container');
        // for (let i = 0; i < itemElements.length; i++) {
        //     const text = itemElements[i].innerText.trim();
        //     if (text.length < 0) continue;

        //     adText.push({
        //         variant: i,
        //         text: text
        //     })
        // }

        return {
            creativeID: info.creativeID,
            advertiserID: info.advertiserID,
            creativeURL: window.location.href,
            advertiserName,
            adFormat,
            // adText,
            iframeInfo: adIframes.length > 0 ? {
                count: adIframes.length,
                sources: Array.from(adIframes).map(f => ({ src: f.src, width: f.width, height: f.height }))
            } : {},
            variations,
            timestamp: new Date().toISOString(),
            status: 'success'
        };
    }, creativeInfo);
}

function extractVideoIframeData(rawText) {
    // Helper function to dynamically build a regex for a specific key
    const extractField = (fieldName) => {
        // This looks for: 'fieldName': 'value' OR "fieldName": "value"
        const regex = new RegExp(`['"]${fieldName}['"]\\s*:\\s*['"]([^'"]+)['"]`, 'i');
        const match = rawText.match(regex);
        return match ? match[1] : null; // Return the captured text, or null if not found
    };

    const headline = extractField('headline');
    const description = extractField('description');
    const callToAction = extractField('call_to_action');
    const advertiser = extractField('channelName'); // In your dump, the brand was under 'channelName'
    const finalUrl = extractField('destination_url') || extractField('visible_url');

    // If we found at least a headline or description, we caught an ad!
    if (headline || description) {
        return [
            headline,
            description,
            callToAction,
            advertiser,
            finalUrl
        ].filter(Boolean); // Remove any null values
    }

    return null; // Return null if this text didn't contain ad data
}

async function extractAdTextFromPage(page) {
    const divs = await page.$$('.creative-container .creative-sub-container');
    const results = [];

    for (var i = 0; i < divs.length; i++) {
        const div = divs[i];
        let hiddenAdData = null;
        let links = []

        // Avoid's policy block message by removing hidden elements before extracting visible text. T
        let visibleText = await page.evaluate((el) => {
            const unwantedElements = el.querySelectorAll('.hidden');
            unwantedElements.forEach(e => e.remove());
            return el.innerText;
        }, div);

        const iframeHandle = await div.$('iframe');

        if (iframeHandle) {
            const frame = await waitToLoadFrame(iframeHandle);

            if (frame) {
                try {
                    // Grab the raw text/script bleed from the iframe
                    const [iframeText, iframeTextRaw] = await frame.evaluate(() => {
                        const rawText = "" + document.documentElement.innerText; // Get the full HTML content of the iframe
                        const clone = document.documentElement.cloneNode(true);

                        // Find all script and style tags inside the iframe
                        const elementsToRemove = clone.querySelectorAll('script, style, noscript');
                        elementsToRemove.forEach(el => el.remove());

                        const elements = Array.from(clone.querySelector('body')?.querySelectorAll('*') || []);

                        const cleanText = elements
                            .filter(node => node.children.length === 0)  // ← only leaf nodes
                            .map(node => node.textContent.trim())
                            .filter(text => text.length > 0)
                            .join('\n');

                        return [cleanText, rawText]; // Return both the cleaned visible text and the raw text
                    });
                    url = await getIframAnchors(frame);                    

                    visibleText += ' ' + iframeText;

                    // Run our Regex extraction!
                    if (iframeTextRaw) {
                        hiddenAdData = extractVideoIframeData(iframeTextRaw);
                    }
                } catch (error) {
                    console.log(`Frame in variant ${i} timed out.`);
                }
            }
        }

        // Clean up the main visible text
        visibleText = visibleText ? visibleText.trim() : "";

        // Only push to results if we actually found something useful
        if (visibleText.length > 0 || hiddenAdData) {
            results.push({
                variant: i,
                visibleText: visibleText,
                hiddenData: hiddenAdData, // This will contain your nice, clean JSON object
                url
            });
        }
    }

    return results;
}

function processIframAnchors(iframeAnchors) {
    if (iframeAnchors.length > 0) {
        const unique = [...new Map(iframeAnchors.map(a => [a.href, a])).values()];
        return {
            isClickable: true,
            targetUrl: (unique.find(a => a.id === 'image-anchor') || unique[0]).href,
            anchorInfo: { count: unique.length, links: unique, source: 'iframe' },
        }
    }

    return {
        isClickable: false,
        targetUrl: null,
        anchorInfo: {},
    }
}

async function getCreativeGrid(creativePage) {
    try {
        return await creativePage.evaluate((baseUrl) => {
            return Array.from(document.querySelectorAll('creative-preview a[href*="/creative/"]'))
                .map(a => {
                    const href = a.getAttribute('href');
                    if (!href) return null;
                    return href.startsWith('http') ? href : `${baseUrl}${href}`;
                })
                .filter(Boolean)
                .filter((v, i, arr) => arr.indexOf(v) === i);
        }, CONFIG.BASE_URL);
    } catch (e) {
        console.warn('Error extracting creative grid URLs:', e);
        return [];
    }
}

// ============ CORE EXTRACTION ============

/**
 * Extract the `adurl` query parameter from a Google redirect URL.
 * Mirrors resolve_target_url() in extract_target_urls.py.
 * Works for both googleadservices.com click-trackers and
 * googlesyndication.com image src URLs that embed the real destination.
 * Returns null when the param is absent or the URL is unparseable.
 */
// TODO: this function does not look good. I since it has some vague 
// logic, I am not changing it, but we may need to refactor it later.
function extractAdUrl(url) {
    if (!url) return null;

    try {
        const params = new URL(url).searchParams;
        const val = params.get('adurl') ?? params.get('adurl ');
        return val ? val.trim() : url;
    } catch {
        return url;
    }
}

function getAppLinkFromAnchors(anchors) {
    if (!anchors || anchors.length === 0) return null;

    return anchors.find(link =>
        APPLE_APP_PATTERNS.some(pattern => link.includes(pattern)) ||
        GOOGLE_APP_PATTERNS.some(pattern => link.includes(pattern))
    );
}

async function extractCreativeData(creativePage, creativeInfo) {
    const imageUrl = await extractImageUrls(creativePage);

    // TODO: strange logind for the tpcSet. Check and change it later.
    const tpcSet = new Set();
    const collectAdLinks = (urls) => urls.filter(Boolean).forEach(u => tpcSet.add(u));

    const domLinks = await creativePage.evaluate(collectLinkUrlsInDocument);
    collectAdLinks(domLinks.filter(isAdLink));

    // Variation 0 = landing page; single-value metadata (targetUrl, adUrl, isClickable) from it only
    const variation0Frame = await getVariationZeroFrame(creativePage);
    const { frameLinks, anchors, landingPageAnchors, iframeTexts, tpcLinks, videoThumbnailUrl } = await extractIframData(creativePage, variation0Frame);
    tpcLinks.forEach(u => tpcSet.add(u));
    collectAdLinks(frameLinks.filter(isAdLink));

    const adText = await extractAdTextFromPage(creativePage);
    const adData = await extractPageData(creativeInfo, creativePage);
    adData.adText = adText;

    // Merge adText: iframe text takes precedence (ad copy lives in the frame, not the main DOM).
    // Fall back to the main-page extraction for text-format ads rendered directly in the Angular DOM.
    // Deduplicate while preserving order.
    // if (iframeTexts.length > 0) {
    //     const merged = [...new Set([...iframeTexts, ...(adData.adText || [])])];
    //     adData.adText = merged;
    // }

    // Use page-extracted image, or fallback to Level 1 image if available
    // For video ads the main-page image extraction rarely finds anything (content is inside the
    // ad iframe). Use the thumbnail extracted from the frame DOM as a fallback.
    adData.imageUrl = imageUrl || (adData.adFormat === 'video' ? videoThumbnailUrl : null) || creativeInfo.imageUrl || null;
    if (adData.adFormat === 'video' && videoThumbnailUrl) adData.videoThumbnailUrl = videoThumbnailUrl;

    // Single-value metadata (targetUrl, adUrl, isClickable) from landing page (variation 0) only
    const anchorsForTarget = (landingPageAnchors && landingPageAnchors.length > 0) ? landingPageAnchors : anchors;
    const landingPageAnchorData = processIframAnchors(anchorsForTarget);
    adData.isClickable = landingPageAnchorData.isClickable;
    adData.targetUrl = landingPageAnchorData.targetUrl;
    adData.adUrl = extractAdUrl(adData.targetUrl);
    // iframeAnchorInfo keeps full list from all variations
    adData.iframeAnchorInfo = processIframAnchors(anchors).anchorInfo;
    
    adData.appLink = getAppLinkFromAnchors(adText.map(a => a.url.map(item => item.adUrl)).flat());

    // tpc_list and creative_grid remain from the whole page
    adData.tpc_list = Array.from(tpcSet);

    adData.creative_grid = await getCreativeGrid(creativePage);

    return adData;
}

module.exports = {
    extractCreativeData,
};
