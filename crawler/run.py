#!/usr/bin/env python3
"""
Python wrapper that uses pandas to load CSV and calls main.js

Usage:
    python run.py ads_asc.csv --max 100
    python run.py ads_asc.csv --max 100 --workers 4
    python run.py ads_desc.csv --filter "advertiser_id == 'AR09188314108603138049'"

    # Resume from progress file(s) after crash (supports multiple files)
    python run.py ads.csv --resume-from "ads_progress_*.json" --workers 4 --screenshots-dir ./images
    python run.py ads.csv --resume-from . --workers 4  # Load all progress files from current directory

    # Store progress files in a specific directory
    python run.py ads.csv --workers 4 --progress-dir ./progress
    python run.py ads.csv --workers 4 --progress-dir /tmp/my_run_progress
"""

import pandas as pd
import subprocess
import argparse
import json
import os
import sys
import time
import glob as glob_module
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import glob
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def monitor_workers(run_start_time, progress_dir, stop_event, interval=15, print_every=10):
    """Background thread: prints worker status every `print_every` new ads processed."""
    search_dir = progress_dir if progress_dir else SCRIPT_DIR
    last_total = 0

    while not stop_event.is_set():
        time.sleep(interval)
        try:
            pattern = os.path.join(search_dir, 'ads_progress_*.json')
            files = [f for f in glob_module.glob(pattern)
                     if os.path.getmtime(f) >= run_start_time - 5]

            worker_counts = {}
            for f in sorted(files):
                name = os.path.basename(f)
                try:
                    data = json.load(open(f))
                    statuses = {}
                    for ad in data:
                        s = ad.get('status', 'unknown')
                        statuses[s] = statuses.get(s, 0) + 1
                    worker_counts[name] = (len(data), statuses)
                except Exception:
                    worker_counts[name] = (0, {})

            total = sum(c for c, _ in worker_counts.values())
            if total - last_total >= print_every or (total > 0 and last_total == 0):
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"\n📊 [{ts}] Worker status ({total} total ads):")
                for name, (count, statuses) in worker_counts.items():
                    label = name.replace('ads_progress_', '').replace('.json', '')
                    ok = statuses.get('success', 0)
                    nf = statuses.get('not_found', 0)
                    err = statuses.get('error', 0)
                    print(f"   {label}: {count} ads  ✅{ok} ⚠️{nf} ❌{err}")
                last_total = total
        except Exception:
            pass


def collect_and_save_progress(final_output, output_files=None, run_start_time=None, progress_dir=None):
    """
    Collect partial results from in-flight progress/chunk files and save them
    to final_output.  Called automatically on Ctrl+C.

    Sources (in priority order):
      1. Parallel chunk files  (/tmp/ads_chunk_*.json)  – one per worker
      2. Incremental progress files (ads_progress_*.json) – written by puppeteer
         every SAVE_INTERVAL ads; only files created after run_start_time are used.
    """
    all_ads = []
    seen_ids = set()
    sources = []

    # ── 1. Chunk files (parallel mode) ──────────────────────────────────────
    for f in (output_files or []):
        if os.path.exists(f):
            try:
                with open(f, 'r') as fp:
                    chunk_ads = json.load(fp)
                new = [a for a in chunk_ads if a.get('creativeID') not in seen_ids]
                all_ads.extend(new)
                seen_ids.update(a.get('creativeID') for a in new)
                sources.append(f"{os.path.basename(f)} ({len(new)} ads)")
                os.remove(f)
            except Exception as e:
                print(f"   ⚠️  Could not read chunk file {f}: {e}")

    # ── 2. Progress files written by main.js ─────────────────────
    search_dir = progress_dir if progress_dir else SCRIPT_DIR
    progress_files = sorted(
        glob_module.glob(os.path.join(search_dir, 'ads_progress_*.json')),
        key=os.path.getmtime
    )
    if run_start_time is not None:
        progress_files = [f for f in progress_files
                          if os.path.getmtime(f) >= run_start_time - 1]

    for f in progress_files:
        try:
            with open(f, 'r') as fp:
                prog_ads = json.load(fp)
            new = [a for a in prog_ads if a.get('creativeID') not in seen_ids]
            all_ads.extend(new)
            seen_ids.update(a.get('creativeID') for a in new)
            if new:
                sources.append(f"{os.path.basename(f)} ({len(new)} new ads)")
        except Exception as e:
            print(f"   ⚠️  Could not read progress file {f}: {e}")

    if all_ads:
        with open(final_output, 'w') as f:
            json.dump(all_ads, f, indent=2)
        print(f"   ✅ Saved {len(all_ads)} ads to: {final_output}")
        if sources:
            print(f"   📦 Sources: {', '.join(sources)}")
    else:
        print("   ⚠️  No partial data found to save.")

    return len(all_ads)


def run_scraper(urls_data, output_file, screenshot_dir, worker_id=None, force_auth=False, progress_dir=None):
    """Run main.js with URLs via temp file"""
    # Write URLs to temp file instead of passing as arg (avoids "Argument list too long")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(urls_data, f)
        temp_file = f.name

    try:
        cmd = ['node', os.path.join(SCRIPT_DIR, 'main.js'),
               '--urls-file', temp_file, '--output', output_file,
               '--screenshots-dir', screenshot_dir]
        
        if worker_id is not None:
            cmd.extend(['--worker-id', str(worker_id)])
        
        if force_auth:
            cmd.append('--force-auth')

        if progress_dir:
            cmd.extend(['--progress-dir', progress_dir])
        
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def run_chunk(chunk_data):
    """Run scraper on a chunk (for parallel processing)"""
    chunk_id, urls, output_file, screenshot_dir, progress_dir = chunk_data
    success = run_scraper(urls, output_file, screenshot_dir, worker_id=chunk_id, progress_dir=progress_dir)
    return chunk_id, output_file, success

def merge_results(output_files, final_output):
    """Merge results from parallel chunks"""
    all_ads = []
    for f in output_files:
        if os.path.exists(f):
            with open(f, 'r') as fp:
                try:
                    all_ads.extend(json.load(fp))
                except:
                    pass
            os.remove(f)
    
    with open(final_output, 'w') as f:
        json.dump(all_ads, f, indent=2)
    return len(all_ads)

def main():
    parser = argparse.ArgumentParser(description='Load CSV with pandas and run puppeteer scraper')
    parser.add_argument('csv_file', nargs='?', help='Path to CSV file')
    parser.add_argument('--max', type=int, default=None, help='Maximum rows to process')
    parser.add_argument('--start', type=int, default=0, help='Start from row (0-indexed)')
    parser.add_argument('--filter', type=str, default=None, help='Pandas query filter')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--screenshots-dir', type=str, default='./results/screenshots', help='Directory to save screenshots (default: ./results/screenshots)')
    parser.add_argument('--setup-profile', action='store_true', help='Setup authentication profile (opens browser for login)')
    parser.add_argument('--retry-failures', type=str, default=None, help='Retry failed ads from previous run (provide path to JSON results)')
    parser.add_argument('--skip', type=int, default=0, help='Skip first N ads (for resuming after crash)')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from progress file(s) - supports wildcards like "ads_progress_*.json" or directory path')
    parser.add_argument('--progress-dir', type=str, default=None, help='Directory to store ads_progress_*.json files (default: script directory)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without running')
    parser.add_argument('--test-urls', type=str, default=None,
                        help='Comma-separated creative page URLs to test (bypasses CSV requirement). '
                             'e.g. --test-urls "https://adstransparency.google.com/advertiser/AR123/creative/CR456,..."')
    
    args = parser.parse_args()

    # Handle test-urls mode: quickly test specific creative URLs without a CSV
    if args.test_urls:
        import re
        urls_raw = [u.strip() for u in args.test_urls.split(',') if u.strip()]
        if not urls_raw:
            print("❌ No valid URLs provided to --test-urls")
            return

        urls_data = []
        for url in urls_raw:
            ar = re.search(r'/advertiser/(AR[\d]+)', url)
            cr = re.search(r'/creative/(CR[\d]+)', url)
            urls_data.append({
                'advertiser_id':            ar.group(1) if ar else None,
                'creative_id':              cr.group(1) if cr else None,
                'creative_page_url':        url,
                'advertiser_disclosed_name': None,
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.output or f'test_{timestamp}.json'
        screenshot_dir = os.path.abspath(args.screenshots_dir)
        os.makedirs(screenshot_dir, exist_ok=True)

        print(f"\n🧪 Test mode: {len(urls_data)} URL(s)")
        for u in urls_data:
            print(f"   {u['creative_page_url']}")

        if not args.dry_run:
            run_scraper(urls_data, output, screenshot_dir, force_auth=args.__dict__.get('force_auth', True))
            print(f"\n✅ Results saved to: {output}")
        else:
            print("\n🔍 Dry run — not executing")
        return

    # Handle setup-profile mode
    if args.setup_profile:
        print("\n🔐 Setting up authentication profile...")
        print("   Browser will open for you to login")
        print("   Close the browser when done\n")
        
        cmd = ['node', os.path.join(SCRIPT_DIR, 'main.js'), '--setup-profile']
        subprocess.run(cmd, cwd=SCRIPT_DIR)
        return
    
    # Handle retry-failures mode
    if args.retry_failures:
        print("\n🔄 Retry Failures Mode")
        print(f"📖 Loading previous results: {args.retry_failures}")
        
        # Check auth profile exists
        profile_path = os.path.join(SCRIPT_DIR, 'browser_profile')
        if not os.path.exists(profile_path):
            print(f"❌ Error: Auth profile not found at {profile_path}")
            print("   Run with --setup-profile first to create an authenticated profile")
            return
        
        # Load previous results
        with open(args.retry_failures, 'r') as f:
            raw_results = json.load(f)
        previous_results = [ad for ad in raw_results if isinstance(ad, dict)]
        if len(previous_results) < len(raw_results):
            print(f"   ⚠️  Skipped {len(raw_results) - len(previous_results)} non-dict entries")
        
        print(f"   Total ads in file: {len(previous_results)}")
        
        # Separate successes and failures
        successes = [ad for ad in previous_results if ad.get('status') == 'success']
        failures = [ad for ad in previous_results if ad.get('status') in ['not_found', 'error']]
        
        print(f"   ✅ Successful ads: {len(successes)}")
        print(f"   ❌ Failed ads to retry: {len(failures)}")
        
        if len(failures) == 0:
            print("\n🎉 No failures to retry!")
            return
        
        # Save failures to pass1.json
        pass1_file = 'pass1.json'
        with open(pass1_file, 'w') as f:
            json.dump(failures, f, indent=2)
        print(f"\n📝 Saved failures to: {pass1_file}")
        
        # Convert to URL format for scraper
        urls_data = []
        for ad in failures:
            urls_data.append({
                'advertiser_id': ad.get('advertiserID'),
                'creative_id': ad.get('creativeID'),
                'creative_page_url': ad.get('creativeURL'),
                'advertiser_disclosed_name': ad.get('advertiserName')
            })
        
        # Apply skip if specified (resume from specific point)
        if args.skip > 0:
            skipped = min(args.skip, len(urls_data))
            urls_data = urls_data[skipped:]
            print(f"\n⏭️  Skipping first {skipped} ads (resuming from ad #{skipped + 1})")
        
        if len(urls_data) == 0:
            print("⚠️  No ads to process after skip")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retry_output = f'ads_retry_{timestamp}.json'
        screenshot_dir = os.path.abspath(args.screenshots_dir)
        os.makedirs(screenshot_dir, exist_ok=True)

        print(f"\n🔐 Retrying with authenticated profile...")
        print(f"📊 Processing {len(urls_data)} failed ads with 1 worker")

        retry_start_time = time.time()
        try:
            # Run with force_auth=True, single worker only
            success = run_scraper(urls_data, retry_output, screenshot_dir, worker_id=None, force_auth=True, progress_dir=args.progress_dir)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted during retry! Saving progress so far...")
            n = collect_and_save_progress(retry_output, run_start_time=retry_start_time, progress_dir=args.progress_dir)
            if n > 0:
                print(f"\n✅ Partial retry results saved: {retry_output}")
            else:
                print("\n   Nothing to save – no ads had been crawled yet.")
            sys.exit(1)

        if not success:
            print("\n❌ Retry failed")
            return
        
        # Load retry results
        print(f"\n📦 Merging results...")
        with open(retry_output, 'r') as f:
            retry_results = json.load(f)
        
        # Merge: original successes + retry results
        final_results = successes + retry_results
        
        # Determine output file name
        final_output = args.output or 'final.json'
        with open(final_output, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Count retry successes
        retry_successes = len([r for r in retry_results if r.get('status') == 'success'])
        retry_failures = len(retry_results) - retry_successes
        
        print(f"\n✅ Merge complete: {final_output}")
        print(f"\n📊 Final Statistics:")
        print(f"   Original successes: {len(successes)}")
        print(f"   Retry successes: {retry_successes}")
        print(f"   Still failed: {retry_failures}")
        print(f"   Total in final.json: {len(final_results)}")
        print(f"\n📁 Files created:")
        print(f"   {pass1_file} - Failed ads extracted for retry")
        print(f"   {retry_output} - Retry results")
        print(f"   {final_output} - Merged final results")
        
        return
    
    # Handle resume mode
    if args.resume_from:
        print("\n🔄 Resume Mode")
        print(f"📖 Loading progress file(s): {args.resume_from}")
        
        # Support wildcards and directory paths
        progress_files = []
        if '*' in args.resume_from:
            # Wildcard pattern
            progress_files = glob.glob(args.resume_from)
        elif os.path.isdir(args.resume_from):
            # Directory - find all ads_progress_*.json files
            progress_files = glob.glob(os.path.join(args.resume_from, 'ads_progress_*.json'))
        elif os.path.isfile(args.resume_from):
            # Single file
            progress_files = [args.resume_from]
        else:
            print(f"❌ Error: No progress files found matching: {args.resume_from}")
            return
        
        if not progress_files:
            print(f"❌ Error: No progress files found")
            return
        
        print(f"   Found {len(progress_files)} progress file(s)")
        
        # Load all progress files
        progress_data = []
        for pf in sorted(progress_files):
            try:
                with open(pf, 'r') as f:
                    data = json.load(f)
                    progress_data.extend(data)
                    print(f"   ✅ {os.path.basename(pf)}: {len(data)} entries")
            except Exception as e:
                print(f"   ⚠️  Skipped {os.path.basename(pf)}: {e}")
        
        print(f"\n   Total entries loaded: {len(progress_data)}")
        
        # Extract crawled creative IDs
        crawled_ids = set()
        success_count = 0
        error_count = 0
        not_found_count = 0
        
        for entry in progress_data:
            creative_id = entry.get('creativeID') or entry.get('creative_id')
            status = entry.get('status', 'unknown')
            
            if creative_id:
                crawled_ids.add(str(creative_id))
                
                if status == 'success':
                    success_count += 1
                elif status == 'error':
                    error_count += 1
                elif status == 'not_found':
                    not_found_count += 1
        
        print(f"\n📊 Progress Statistics:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   ⚠️  Not Found: {not_found_count}")
        print(f"   Total unique IDs crawled: {len(crawled_ids)}")
        
        # CSV file required for resume mode
        if not args.csv_file:
            print("\n❌ Error: CSV file required for resume mode")
            print("   Usage: python run.py <csv_file> --resume-from <progress_pattern>")
            print("   Examples:")
            print('     python run.py ads.csv --resume-from "ads_progress_*.json" --workers 4')
            print('     python run.py ads.csv --resume-from . --workers 4')
            return
        
        # Load and filter CSV
        print(f"\n📖 Loading original CSV: {args.csv_file}")
        df = pd.read_csv(args.csv_file)
        original_count = len(df)
        print(f"   Total rows in CSV: {original_count}")
        
        # Apply filters if specified
        if args.filter:
            print(f"🔍 Filter: {args.filter}")
            df = df.query(args.filter)
            print(f"   After filter: {len(df)}")
        
        # Check for creative_id column
        if 'creative_id' not in df.columns:
            print(f"❌ Error: 'creative_id' column not found in CSV")
            print(f"   Available columns: {list(df.columns)}")
            return

        # Convert creative_id to string for comparison
        df['creative_id'] = df['creative_id'].astype(str)

        # Generate creative_page_url if missing
        if 'creative_page_url' not in df.columns:
            print("   ℹ️  No 'creative_page_url' column — generating from advertiser_id + creative_id")
            df['creative_page_url'] = (
                'https://adstransparency.google.com/advertiser/'
                + df['advertiser_id'].astype(str)
                + '/creative/'
                + df['creative_id'].astype(str)
            )
        
        # Filter out already crawled entries
        df_remaining = df[~df['creative_id'].isin(crawled_ids)]
        remaining_count = len(df_remaining)
        crawled_count = len(df) - remaining_count
        
        print(f"\n📊 Filtering Results:")
        print(f"   Already crawled: {crawled_count}")
        print(f"   Remaining to crawl: {remaining_count}")
        if len(df) > 0:
            print(f"   Progress: {crawled_count}/{len(df)} ({100*crawled_count/len(df):.1f}%)")
        
        if remaining_count == 0:
            print("\n🎉 All entries already crawled!")
            return
        
        # Use filtered dataframe
        df = df_remaining
        
        # Continue with normal processing below
        print(f"\n✅ Resuming with {len(df)} remaining rows")
        
        # Show first few URLs
        if len(df) > 0:
            print(f"\n📋 First 3 remaining:")
            for url in df['creative_page_url'].head(3):
                print(f"   {url}")
        
        # Prepare for scraping (skip to the urls_data preparation)
        # Set flag to indicate we're in resume mode
        resume_mode = True
    else:
        resume_mode = False
    
    # CSV file required for normal mode
    if not args.csv_file and not resume_mode:
        parser.print_help()
        return
    
    # Load CSV (skip if already loaded in resume mode)
    if not resume_mode:
        print(f"📖 Loading: {args.csv_file}")
        df = pd.read_csv(args.csv_file)
        print(f"   Total rows: {len(df)}")

        # Generate creative_page_url if missing
        if 'creative_page_url' not in df.columns:
            print("   ℹ️  No 'creative_page_url' column — generating from advertiser_id + creative_id")
            df['creative_page_url'] = (
                'https://adstransparency.google.com/advertiser/'
                + df['advertiser_id'].astype(str)
                + '/creative/'
                + df['creative_id'].astype(str)
            )

        # Filter
        if args.filter:
            print(f"🔍 Filter: {args.filter}")
            df = df.query(args.filter)
            print(f"   After filter: {len(df)}")

        # Slice
        if args.start > 0 or args.max:
            end = args.start + (args.max if args.max else len(df))
            df = df.iloc[args.start:end]

        print(f"\n✅ Processing {len(df)} rows with {args.workers} worker(s)")

        if len(df) > 0:
            print(f"\n📋 First 3:")
            for url in df['creative_page_url'].head(3):
                print(f"   {url}")
    
    if args.dry_run or len(df) == 0:
        print("\n🔍 Dry run" if args.dry_run else "⚠️ No rows")
        return
    
    # Prepare URL data
    cols = ['advertiser_id', 'creative_id', 'creative_page_url']
    if 'advertiser_disclosed_name' in df.columns:
        cols.append('advertiser_disclosed_name')
    urls_data = df[cols].to_dict('records')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = args.output or f'ads_{timestamp}.json'

    # Setup screenshot directory
    screenshot_dir = os.path.abspath(args.screenshots_dir)
    os.makedirs(screenshot_dir, exist_ok=True)
    print(f"\n📸 Screenshots will be saved to: {screenshot_dir}")

    # Setup progress directory
    progress_dir = args.progress_dir
    if progress_dir:
        progress_dir = os.path.abspath(progress_dir)
        os.makedirs(progress_dir, exist_ok=True)
        print(f"\n💾 Progress files will be saved to: {progress_dir}")

    # Track chunk files so the Ctrl+C handler can find them
    output_files = []
    run_start_time = time.time()

    try:
        if args.workers == 1:
            # Single worker
            print(f"\n🚀 Running scraper...")
            run_scraper(urls_data, final_output, screenshot_dir, progress_dir=progress_dir)
        else:
            # Parallel - split into chunks
            chunk_size = math.ceil(len(urls_data) / args.workers)
            chunks = []

            for i in range(args.workers):
                start = i * chunk_size
                end = min(start + chunk_size, len(urls_data))
                if start >= len(urls_data):
                    break

                chunk_output = f'/tmp/ads_chunk_{i}_{timestamp}.json'
                chunks.append((i, urls_data[start:end], chunk_output, screenshot_dir, progress_dir))
                output_files.append(chunk_output)
                print(f"   Chunk {i+1}: {end - start} URLs")

            print(f"\n🚀 Running {len(chunks)} parallel scrapers...")

            stop_monitor = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_workers,
                args=(run_start_time, progress_dir, stop_monitor),
                daemon=True,
            )
            monitor_thread.start()

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_chunk, c): c[0] for c in chunks}
                for future in as_completed(futures):
                    cid, out, ok = future.result()
                    print(f"   {'✅' if ok else '❌'} Chunk {cid + 1}")

            stop_monitor.set()
            monitor_thread.join(timeout=2)

            print(f"\n📦 Merging...")
            total = merge_results(output_files, final_output)
            print(f"   {total} ads merged")

        print(f"\n✅ Done: {final_output}")
        print(f"📸 Screenshots: {screenshot_dir}")

        # If resume mode, provide merge instructions
        if resume_mode and args.resume_from:
            print(f"\n💡 To merge with original progress file:")
            print(f"   1. Load progress: {args.resume_from}")
            print(f"   2. Load new results: {final_output}")
            print(f"   3. Combine both JSON arrays and save to final.json")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving progress so far...")
        n = collect_and_save_progress(
            final_output,
            output_files=output_files,
            run_start_time=run_start_time,
            progress_dir=progress_dir,
        )
        if n > 0:
            print(f"\n✅ Partial results saved: {final_output}")
        else:
            print("\n   Nothing to save – no ads had been crawled yet.")
        sys.exit(1)

if __name__ == '__main__':
    main()
