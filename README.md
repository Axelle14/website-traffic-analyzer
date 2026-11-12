# Website Traffic Analyzer

A small, dependency-free Python script that summarizes website traffic from a
CSV file and writes a report to `output/results.txt`.

## Project structure

```text
Website_Traffic_Analyzer/
├── data/
│   └── website_traffic.csv
├── src/
│   └── website_traffic_analyzer.py
├── output/
│   └── results.txt
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.8 or newer
- No third-party packages

## Run

From the project directory:

```bash
python src/website_traffic_analyzer.py
```

The script reads `data/website_traffic.csv`, prints the report, and writes the
same report to `output/results.txt`.

## CSV format

The CSV must include these columns:

| Column | Description |
| --- | --- |
| `date` | Date for the traffic record |
| `page` | Page path, such as `/products` |
| `visitors` | Non-negative visitor count |
| `page_views` | Non-negative page-view count |
| `bounce_rate` | Bounce rate as a percentage from 0 to 100 |
| `avg_session_duration_seconds` | Non-negative average session duration in seconds |

The report includes totals, visitor-weighted bounce rate and session duration,
and the page with the most page views. Replace the sample rows with your own
data while keeping the same column names.
