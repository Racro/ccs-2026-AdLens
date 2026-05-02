const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

function saveResult(outputFile, result) {
    fs.appendFileSync(outputFile, JSON.stringify(result) + "\n");
}

function categorizeAppFromContent(data) {
    const text = (
        data.title + " " +
        data.description + " " +
        data.category
    ).toLowerCase();

    const rules = [
        {
            name: "Phone cleaner / optimizer",
            keywords: ["cleaner", "junk", "cache", "boost", "optimizer", "storage"]
        },
        {
            name: "File recovery",
            keywords: ["recover", "recovery", "restore", "deleted"]
        },
        {
            name: "PDF / document reader",
            keywords: ["pdf", "reader", "document", "docx", "viewer"]
        },
        {
            name: "Phone tracker",
            keywords: ["tracker", "location", "gps", "locator"]
        }
    ];

    let bestCategory = "Other";
    let bestScore = 0;

    for (const rule of rules) {
        let score = 0;

        for (const kw of rule.keywords) {
            if (text.includes(kw)) score++;
        }

        if (score > bestScore) {
            bestScore = score;
            bestCategory = rule.name;
        }
    }

    return bestScore > 0 ? bestCategory : "Other";
}

async function scrapeGooglePlay(page, url) {
    await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });

    for (let i = 0; i < 5; i++) {
        await page.evaluate(() => window.scrollBy(0, window.innerHeight));
        await new Promise(r => setTimeout(r, 1500));
    }

    const isUnavailable = await page.evaluate(() =>
        document.body.innerText.includes("We're sorry")
    );

    if (isUnavailable) {
        return {
            appLink: url,
            reviews: [],
            overallRating: null,
            deleted: true,
            downloads: null,
            publisherData: null,
            category: null
        };
    }

    const overallRating = await page.evaluate(() => {
        const el = document.querySelector('.jILTFe');
        return el ? parseFloat(el.innerText) : null;
    });

    try {
        const [button] = await page.$x("//span[contains(text(),'See all reviews')]/ancestor::button");

        if (button) {
            await button.click();
            await page.waitForSelector('div[role="dialog"]', { timeout: 5000 });
        }
    } catch { }

    const reviews = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('div.h3YV2d'))
            .map(el => el.innerText.trim())
            .filter(Boolean)
            .slice(0, 20);
    });

    const downloads = await page.evaluate(() => {
        const items = Array.from(document.querySelectorAll('div.wVqUob'));

        for (const item of items) {
            const label = item.querySelector('.g1rdde')?.innerText;
            if (label && label.toLowerCase().includes('download')) {
                return item.querySelector('.ClM7O')?.innerText || null;
            }
        }

        return null;
    });

    const publisherData = await page.evaluate(() => {
        const link = document.querySelector(
            'a[href^="/store/apps/developer"], a[href^="/store/apps/dev"]'
        );

        if (!link) {
            return { name: null, developerId: null };
        }

        const name = link.querySelector('span')?.innerText.trim() || null;

        const href = link.getAttribute('href');

        let developerId = null;

        if (href) {
            const match = href.match(/id=([^&]+)/);
            if (match) developerId = decodeURIComponent(match[1]);
        }

        return { name, developerId };
    });

    const pageData = await page.evaluate(() => {
        const title =
            document.querySelector('h1[itemprop="name"] span')?.innerText || "";

        const description =
            document.querySelector('div[data-g-id="description"]')?.innerText || "";

        const category =
            document.querySelector('a[itemprop="genre"]')?.innerText || "";

        return { title, description, category };
    });

    const category = categorizeAppFromContent(pageData);

    return {
        appLink: url,
        reviews,
        overallRating,
        deleted: false,
        downloads,
        publisherData,
        category
    };
}

function extractAppId(url) {
    const match = url.match(/id(\d+)/);
    return match ? match[1] : null;
}

async function fetchAppleReviews(appId) {
    // `https://itunes.apple.com/rss/customerreviews/id6742165758/json`;
    const url = `https://itunes.apple.com/rss/customerreviews/id=${appId}/json`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        const entries = data.feed?.entry || [];

        // First entry is app metadata, skip it
        const reviews = entries.slice(1).map(e => e.content?.label).filter(Boolean);

        const overallRating = parseFloat(
            data.feed?.entry?.[0]?.["im:rating"]?.label || null
        );

        return {
            reviews: reviews.slice(0, 20),
            overallRating
        };

    } catch (err) {
        return {
            reviews: [],
            overallRating: null
        };
    }
}

async function fetchAppleMetadata(appId) {
    const url = `https://itunes.apple.com/lookup?id=${appId}`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        const app = data.results?.[0];

        if (!app) return {};

        return {
            name: app.trackName,
            developer: app.sellerName,
            genre: app.primaryGenreName,
            price: app.formattedPrice,
            ratingCount: app.userRatingCount
        };

    } catch {
        return {};
    }
}

async function scrapeAppStore(url) {
    const appId = extractAppId(url);

    if (!appId) {
        return {
            appLink: url,
            reviews: [],
            overallRating: null,
            metadata: {},
            deleted: true
        };
    }

    const [reviewData, metadata] = await Promise.all([
        fetchAppleReviews(appId),
        fetchAppleMetadata(appId)
    ]);

    return {
        appLink: url,
        reviews: reviewData.reviews,
        overallRating: reviewData.overallRating,
        metadata,
        deleted: false
    };
}

function parseCliArgs() {
    const [inputPathArg, outputPathArg] = process.argv.slice(2);

    if (!inputPathArg || !outputPathArg) {
        console.error("Usage: node app.js <input_json_path> <output_jsonl_path>");
        process.exit(1);
    }

    const inputFile = path.resolve(inputPathArg);
    const outputFile = path.resolve(outputPathArg);

    if (!fs.existsSync(inputFile)) {
        console.error(`Input file not found: ${inputFile}`);
        process.exit(1);
    }

    fs.mkdirSync(path.dirname(outputFile), { recursive: true });

    return { inputFile, outputFile };
}

async function scrapeAll(inputFile, outputFile) {
    const browser = await puppeteer.launch({
        headless: false,
        args: ["--no-sandbox"]
    });

    const page = await browser.newPage();

    const links = [...new Set(JSON.parse(fs.readFileSync(inputFile, "utf-8")))];

    for (const link of links) {
        console.log("Scraping:", link);

        let result;

        try {
            if (link.includes("play.google.com")) {
                result = await scrapeGooglePlay(page, link);
            } else {
                result = await scrapeAppStore(link);
            }
        } catch (e) {
            result = {
                appLink: link,
                reviews: [],
                overallRating: null,
                deleted: true
            };
        }

        // ✅ SAVE IMMEDIATELY
        saveResult(outputFile, result);

        console.log("Saved:", link);

        await delay(3000);
    }

    await browser.close();
}

const { inputFile, outputFile } = parseCliArgs();
scrapeAll(inputFile, outputFile);
