# Violations

Sample confirmed-violation ad images used as examples in the paper. Each subfolder contains 200 ad screenshots that were classified as violations by the AdLens ensemble pipeline and manually verified.

## Structure

| Folder | Violation Type | Description |
|---|---|---|
| `deceptive_claims/` | Deceptive Claim | Ads making false device-state, fake recovery, or financial bait claims |
| `misleading_ad_design/` | Misleading Ad Design | CTA-only ads with no identifiable advertiser |
| `scareware/` | Scareware | Ads with assertive threat or panic-inducing claims |

## Filename Format

Images follow the Google Ad Transparency Center creative ID scheme:

```
<creative_id>-v<version>.png
```

For example, `CR00173184909714653185-v1.png` is creative 1 of ad ID `CR00173184909714653185`.
