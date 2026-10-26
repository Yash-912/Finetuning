| Set | Metric | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Test split (in-distribution) | Attack recall | 94.8% (699/737) | 95.1% (701/737) | 96.2% (709/737) | 94.8% (699/737) |
| | Benign FPR | 0.4% (6/1460) | 0.3% (5/1460) | 0.3% (5/1460) | 0.3% (5/1460) |
| Hand-written direct attacks, 47 items | Attack recall | 93.0% (40/43) | 86.1% (37/43) | 83.7% (36/43) | 90.7% (39/43) |
| | Benign FPR | 25.0% (1/4) | 0.0% (0/4) | 0.0% (0/4) | 0.0% (0/4) |
| Hand-written set, 112 prompts | Attack recall | 96.3% (52/54) | 96.3% (52/54) | 92.6% (50/54) | 92.6% (50/54) |
| | Benign FPR | 6.9% (4/58) | 0.0% (0/58) | 0.0% (0/58) | 0.0% (0/58) |
| BIPIA test split (indirect: email, table, code) | Attack recall | 14.7% (1214/8250) | 9.8% (811/8250) | 93.4% (7703/8250) | 86.6% (7144/8250) |
| | Benign FPR | 1.5% (3/200) | 1.5% (3/200) | 13.0% (26/200) | 0.5% (1/200) |
| Indirect set, 240 items | Attack recall | 25.0% (30/120) | 16.7% (20/120) | 29.2% (35/120) | 27.5% (33/120) |
| | Benign FPR | 6.7% (8/120) | 5.0% (6/120) | 4.2% (5/120) | 3.3% (4/120) |
| Long-document indirect set, 120 items (not ranked) | Attack recall | 3.3% (2/60) | 5.0% (3/60) | 21.7% (13/60) | 15.0% (9/60) |
| | Benign FPR | 1.7% (1/60) | 3.3% (2/60) | 6.7% (4/60) | 1.7% (1/60) |
| Held-out benign set, 200 prompts | Benign FPR | 4.5% (9/200) | 0.0% (0/200) | 0.0% (0/200) | 0.0% (0/200) |
| **Ranking score (mean Youden J)** | | 0.567 | 0.594 | 0.755 | **0.776** |
| Pooled FPR, all benign prompts | | 1.52% | 0.76% | 1.90% | 0.52% |
