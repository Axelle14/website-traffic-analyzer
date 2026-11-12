import csv
from collections import Counter, defaultdict
import heapq
import time
from datetime import datetime
from pathlib import Path


def load_data(filename):
    """Read website traffic records from a CSV file."""

    records = []

    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            records.append(row)

    return records


def inspect_data(records):
    """Display basic information about the dataset."""

    print("\n========== DATASET INSPECTION ==========")

    print("Total records:", len(records))

    if records:
        print("Columns:", list(records[0].keys()))

        print("\nFirst record:")
        print(records[0])

        print("\nLast record:")
        print(records[-1])


def clean_data(records):
    """Validate records, convert types, remove duplicates.
    Returns (clean_records, stats).

    One row = one HTTP request, identified by request_id, made by a
    client IP at a given time to a path, which returned a status code
    after response_ms milliseconds.
    """
    stats = {"read": len(records), "missing": 0, "invalid": 0,
             "duplicate": 0, "kept": 0}
    required = ["request_id", "timestamp", "client_ip",
                "path", "status_code", "response_ms"]
    seen_ids = set()
    clean = []

    for row in records:
        # 1. Missing values: any field absent or blank
        if any(row.get(f) is None or row[f].strip() == "" for f in required):
            stats["missing"] += 1
            continue

        # 2. Invalid values: wrong types or out-of-range numbers
        try:
            request_id = int(row["request_id"])
            status = int(row["status_code"])
            response_ms = int(row["response_ms"])
            ts = datetime.fromisoformat(row["timestamp"].strip())
        except ValueError:
            stats["invalid"] += 1
            continue
        if not (100 <= status <= 599) or response_ms < 0:
            stats["invalid"] += 1
            continue

        # 3. Duplicates: same request_id seen before
        if request_id in seen_ids:
            stats["duplicate"] += 1
            continue
        seen_ids.add(request_id)

        # 4. Keep a cleaned copy with proper types
        clean.append({
            "request_id": request_id,
            "timestamp": ts,
            "client_ip": row["client_ip"].strip(),
            "path": row["path"].strip(),
            "status_code": status,
            "response_ms": response_ms,
        })

    stats["kept"] = len(clean)
    return clean, stats


# ======================================================================
# MAPREDUCE PIPELINE
# ======================================================================

def map_record(row):
    """MAP stage: turn ONE cleaned row into several (key, value) pairs.

    Example: /login, status 503, 14:30 produces
        (("page", "/login"), 1)
        (("status", 503), 1)
        (("hour", 14), 1)
        (("error_page", "/login"), 1)     <- only because 503 >= 400
    The first part of each key says WHICH question the pair answers.
    """
    pairs = [
        (("page", row["path"]), 1),
        (("status", row["status_code"]), 1),
        (("hour", row["timestamp"].hour), 1),
    ]
    if row["status_code"] >= 400:
        pairs.append((("error_page", row["path"]), 1))
    return pairs


def shuffle(mapped_pairs):
    """GROUP / SHUFFLE stage: collect all values that share the same key.

    Input : [(("page", "/home"), 1), (("page", "/home"), 1), ...]
    Output: {("page", "/home"): [1, 1, ...], ...}
    On a real cluster this is the step that sends every pair with the
    same key to the same machine.
    """
    groups = defaultdict(list)
    for key, value in mapped_pairs:
        groups[key].append(value)
    return groups


def reduce_group(key, values):
    """REDUCE stage: combine the list of values for ONE key into a result.
    Here the result is the total count."""
    return key, sum(values)


def run_mapreduce(clean, verbose=True):
    """Run map -> shuffle -> reduce and split results by key type."""
    # MAP: every row produces several pairs
    mapped = []
    for row in clean:
        mapped.extend(map_record(row))

    # SHUFFLE: group the values by key
    groups = shuffle(mapped)

    # REDUCE: one result per key
    reduced = [reduce_group(key, values) for key, values in groups.items()]

    # Split results by the first part of the key
    results = {"page": {}, "status": {}, "hour": {}, "error_page": {}}
    for (kind, name), total in reduced:
        results[kind][name] = total

    if verbose:
        print("\n========== MAPREDUCE SUMMARY ==========")
        print("Rows mapped:", len(clean))
        print("Key-value pairs produced by mapper:", len(mapped))
        print("Distinct keys after shuffle:", len(groups))
    return results, mapped, groups


def top_k(counts, k=5):
    """ADDITIONAL TECHNIQUE - Top-K using a heap.

    heapq.nlargest keeps only the k best items while scanning, instead of
    sorting everything. That costs about O(n log k) rather than O(n log n),
    which matters when there are millions of distinct keys.
    """
    return heapq.nlargest(k, counts.items(), key=lambda item: item[1])


def error_rates(page_counts, error_counts):
    """ADDITIONAL TECHNIQUE - error rate per page = errors / requests.

    Raw error counts favour busy pages; the rate shows which page is
    actually the most unreliable. Returns [(page, rate, errors, requests)]
    sorted from worst to best.
    """
    rates = []
    for page, total in page_counts.items():
        errors = error_counts.get(page, 0)
        rates.append((page, errors / total, errors, total))
    return sorted(rates, key=lambda r: r[1], reverse=True)


def show_results(results):
    """Print the answers to the four analysis tasks."""
    print("\n========== 1. TOP 5 MOST VISITED PAGES ==========")
    for rank, (page, n) in enumerate(top_k(results["page"]), 1):
        print(f"{rank}. {page:<12} {n}")

    print("\n========== 2. HTTP STATUS CODE COUNTS ==========")
    for code in sorted(results["status"]):
        print(f"{code}: {results['status'][code]}")

    print("\n========== 3. PAGES GENERATING THE MOST ERRORS ==========")
    for page, n in top_k(results["error_page"], k=len(results["error_page"])):
        print(f"{page:<12} {n}")

    print("\n========== EXTRA: ERROR RATE PER PAGE ==========")
    print(f"{'page':<12} {'rate':>7} {'errors':>7} {'requests':>9}")
    for page, rate, errors, total in error_rates(results["page"], results["error_page"]):
        print(f"{page:<12} {rate:>6.1%} {errors:>7} {total:>9}")

    print("\n========== 4. TRAFFIC BY HOUR ==========")
    for hour in sorted(results["hour"]):
        print(f"{hour:02d}:00  {results['hour'][hour]}")

def verify_results(clean, results):
    """VERIFICATION: recompute every result a different way and compare.

    Check 1 uses collections.Counter (plain Python, no MapReduce).
    Check 2 uses Pandas groupby, if Pandas is installed.
    assert stops the program with an error if any number differs.
    """
    print("\n========== VERIFICATION ==========")

    # Check 1: Counter
    pages = Counter(r["path"] for r in clean)
    statuses = Counter(r["status_code"] for r in clean)
    errors = Counter(r["path"] for r in clean if r["status_code"] >= 400)
    hours = Counter(r["timestamp"].hour for r in clean)

    assert dict(pages) == results["page"], "page counts differ"
    assert dict(statuses) == results["status"], "status counts differ"
    assert dict(errors) == results["error_page"], "error counts differ"
    assert dict(hours) == results["hour"], "hourly counts differ"
    print("Check 1 (Counter): all four results match MapReduce.")

    # Check 2: Pandas (optional)
    try:
        import pandas as pd
    except ImportError:
        print("Check 2 (Pandas): skipped, Pandas is not installed.")
        return

    df = pd.DataFrame(clean)
    df["hour"] = df["timestamp"].dt.hour
    assert df["path"].value_counts().to_dict() == results["page"]
    assert df["status_code"].value_counts().to_dict() == results["status"]
    assert df[df["status_code"] >= 400].groupby("path").size().to_dict() == results["error_page"]
    assert df.groupby("hour").size().to_dict() == results["hour"]
    print("Check 2 (Pandas): all four results match MapReduce.")

    # Hand-check of one value, as you would explain it in a demo
    login_rows = sum(1 for r in clean if r["path"] == "/login")
    assert login_rows == results["page"]["/login"]
    print(f"Spot check: counted /login rows by hand = {login_rows}, "
          f"reducer says {results['page']['/login']}.")

def performance_test(clean, multipliers=(1, 10, 100, 1000)):
    """ADDITIONAL TECHNIQUE - performance measurement.

    Repeats the cleaned data N times to simulate bigger datasets and times
    the full map -> shuffle -> reduce pipeline with time.perf_counter().
    Also reports how many key-value pairs the shuffle has to hold, because
    that is what uses memory.
    """
    print("\n========== PERFORMANCE MEASUREMENT ==========")
    print(f"{'rows':>10} {'pairs held':>12} {'seconds':>9} {'rows/sec':>11}")
    for m in multipliers:
        big = clean * m
        start = time.perf_counter()
        _, mapped, _ = run_mapreduce(big, verbose=False)
        elapsed = time.perf_counter() - start
        print(f"{len(big):>10,} {len(mapped):>12,} {elapsed:>9.3f} {len(big) / elapsed:>11,.0f}")

def run_with_combiner(clean, chunk_size=1000):
    """ADDITIONAL TECHNIQUE - combiner + chunked processing.

    The data is split into chunks. Inside each chunk the mapper output is
    summed locally (the COMBINER), so a chunk sends only ONE pair per key to
    the shuffle instead of one pair per row. The shuffle and reducer then
    work on these much smaller partial results.

    Returns (results, pairs_sent_to_shuffle).
    """
    groups = defaultdict(list)
    pairs_sent = 0

    for start in range(0, len(clean), chunk_size):
        chunk = clean[start:start + chunk_size]

        # MAP + COMBINE (local to this chunk)
        local = defaultdict(int)
        for row in chunk:
            for key, value in map_record(row):
                local[key] += value

        # Only the partial sums are sent to the shuffle
        for key, partial_total in local.items():
            groups[key].append(partial_total)
            pairs_sent += 1

    # REDUCE: sum the partial totals for each key
    results = {"page": {}, "status": {}, "hour": {}, "error_page": {}}
    for key, values in groups.items():
        (kind, name), total = reduce_group(key, values)
        results[kind][name] = total

    return results, pairs_sent


def combiner_comparison(clean, results, multiplier=1000):
    """Compare the plain pipeline with the combiner version on big data."""
    big = clean * multiplier
    print("\n========== COMBINER COMPARISON ==========")
    print(f"Dataset size: {len(big):,} rows")

    start = time.perf_counter()
    plain_results, plain_pairs, _ = run_mapreduce(big, verbose=False)
    plain_time = time.perf_counter() - start

    start = time.perf_counter()
    comb_results, comb_pairs = run_with_combiner(big)
    comb_time = time.perf_counter() - start

    # The combiner must not change any answer
    assert plain_results == comb_results, "combiner changed the results!"
    # Sanity check against the original small-data results
    for kind in results:
        for name, total in results[kind].items():
            assert comb_results[kind][name] == total * multiplier

    print(f"{'method':<22} {'pairs to shuffle':>17} {'seconds':>9}")
    print(f"{'plain MapReduce':<22} {len(plain_pairs):>17,} {plain_time:>9.3f}")
    print(f"{'with combiner':<22} {comb_pairs:>17,} {comb_time:>9.3f}")
    print(f"Pairs reduced by a factor of {len(plain_pairs) / comb_pairs:,.0f}; "
          "results identical.")


def main():

    # Locate the CSV file
    project_folder = Path(__file__).resolve().parent.parent
    filename = project_folder / "data" / "02_website_traffic.csv"

    # Load dataset
    records = load_data(filename)

    # Inspect dataset
    inspect_data(records)

    # Clean dataset
    clean, stats = clean_data(records)

    print("\n========== CLEANING REPORT ==========")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("Sample cleaned record:", clean[0])

    # Map -> shuffle -> reduce
    results, mapped, groups = run_mapreduce(clean)

    # Show one record being mapped (useful for the demo)
    print("\nPairs produced by the mapper for the first row:")
    for pair in map_record(clean[0]):
        print("  ", pair)

    show_results(results)

    # Verify against independent calculations
    verify_results(clean, results)

    # Measure how the pipeline scales
    performance_test(clean)

    # Show the combiner fix
    combiner_comparison(clean, results)


if __name__ == "__main__":
    main()