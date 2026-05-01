const fs = require('fs');
const path = require('path');

function saveADs(ads, outputFile) {
    fs.writeFileSync(outputFile, JSON.stringify(ads, null, 2));
    console.log(`Saved ${ads.length} ads to ${outputFile}`);
}

function parseJsonURL(jsonObject) {
    return ({
        advertiserID: jsonObject.advertiser_id || null,
        creativeID: jsonObject.creative_id || null,
        advertiserName: jsonObject.advertiser_disclosed_name || null,
        relativeURL: (jsonObject.creative_page_url || '').replace('https://adstransparency.google.com', ''),
        imageUrl: null
    })
}

function loadUrlFile(filePath) {
    // URL list mode from file (from pandas via run.py)
    const urlsData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    return urlsData.map(r => parseJsonURL(r));
}

function loadUrlArg(urlsArg) {
    // URL list mode from argument (backwards compatibility)
    const urlsData = JSON.parse(urlsArg);

    if (!!urlsData && !Array.isArray(urlsData)) {
        throw new Error('Invalid --urls argument. Must be a JSON string of an array of URL objects.');
    }

    return urlsData.map(r => parseJsonURL(r));
}

function getScreenshotDir(baseDir, workerId = null) {
    if (!!workerId) {
        return path.join(baseDir, `worker_${workerId}`);
    }

    return baseDir;
}

function createScreenshotDir(dir) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`📸 Screenshots will be saved to: ${dir}`);

    return dir;
}

// Stable ID for this process — ensures all progress files from one run share the same prefix.
const RUN_ID = Date.now();

// One file per worker (overwritten each save). workerId may be null for single-worker (main).
function writeProgress(ads, progressDir, workerId = null) {
    if (!progressDir) return;

    fs.mkdirSync(progressDir, { recursive: true });

    const suffix = workerId !== undefined && workerId !== null ? `worker_${workerId}` : 'main';
    const fileName = `ads_progress_${RUN_ID}_${suffix}.json`;
    const progressPath = path.join(progressDir, fileName);

    try {
        fs.writeFileSync(progressPath, JSON.stringify(ads, null, 2));
        console.log(`💾 Progress saved to ${progressPath}`);
    } catch (err) {
        console.error(`❌ Failed to save progress to ${progressPath}: ${err.message}`);
    }
}

module.exports = {
    saveADs,
    parseJsonURL,
    loadUrlFile,
    loadUrlArg,
    createScreenshotDir,
    getScreenshotDir,
    writeProgress
};