# ChronoMiner Daily Extraction Runs

Production extraction of 224 in-scope Michelin Guide PDFs
(7 series, 1975-2023) using gpt-5.4-mini via ChronoMiner.

## Setup summary

- Schema: `MichelinGuidesLight`
- Input: `data/raw/chronominer/production/{series}/`
- Output: next to input (input_paths_is_output_path = true)
- Context text: `production/{series}_extract_context.txt`
  (auto-resolved via folder hierarchy)
- Context image: `production/{series}_extract_context.png`
  (symbol table; auto-resolved)
- Reasoning: medium (1975-2020), low (2021+)
- Daily token limit: 10M per key
- Service tier: flex (synchronous)

## Daily workflow

### 1. Set the target series

Edit `ChronoMiner/config/paths_config.yaml`, section
`schemas_paths.MichelinGuidesLight.input` (and `output`):

```yaml
MichelinGuidesLight:
  input: "C:/Users/pagoetz/PycharmProjects/NetworksOfTaste/data/raw/chronominer/production/schweiz"
  output: "C:/Users/pagoetz/PycharmProjects/NetworksOfTaste/data/raw/chronominer/production/schweiz"
```

### 2. Set reasoning effort

Edit `ChronoMiner/config/model_config.yaml`:

```yaml
extraction_model:
  reasoning:
    effort: medium   # 1975-2020 editions
    # effort: low    # 2021+ editions (France only)
```

### 3. Launch ChronoMiner (interactive mode)

```powershell
cd C:\Users\pagoetz\PycharmProjects\ChronoMiner
uv run python main/process_text_files.py
```

### 4. Walk through the prompts

| Prompt | Selection |
|---|---|
| Schema | MichelinGuidesLight |
| Processing mode | Synchronous |
| Resume mode | Skip / resume partial |
| Context mode | Automatic |
| Context image | Yes |
| Image detail | High |
| Chunk/page range | All |
| Input selection | Process all files in folder |
| Confirm | Yes |

### 5. Run again with second API key

Swap `OPENAI_API_KEY` in your environment (PyCharm run config)
and repeat steps 3-4. Resume mode skips already-completed files.

### 6. Next day

Repeat from step 3. Resume mode picks up where yesterday left
off.

## Reasoning effort transition

Once all 1975-2020 editions in a series are done, switch
`reasoning.effort` to `low` in model_config.yaml before
processing 2021+ editions. Only France has editions past 2020
(2021, 2022, 2023).

## Suggested series order (smallest first)

| # | Series | Guides | Content pages |
|---|---|---|---|
| 1 | schweiz | 17 | 6,032 |
| 2 | benelux | 20 | 10,207 |
| 3 | portugal_spain | 32 | 20,476 |
| 4 | britain | 37 | 29,510 |
| 5 | deutschland | 33 | 39,869 |
| 6 | italy | 36 | 36,691 |
| 7 | france | 49 | 93,188 |

## Deadline strategy (07.06.2025)

Use free-tier tokens daily until the deadline. On the last day,
finish remaining guides either:

- in flex synchronous mode (same as daily runs), or
- in batch mode (`--batch` flag, 50% cost reduction, 24h
  turnaround) -- note: batch mode does NOT support context
  images.

## Token budget reference

Test run (50 pages, benelux_1982): 330K tokens = 3.3% of 10M.
Rough estimate: ~6,600 tokens per page (input + output with
medium reasoning). At 10M tokens/key/day, each key processes
roughly 1,500 pages per day.

## Per-guide content page ranges

Source: `src/michelin_panel/edition_metadata.json`
(`edition_page_ranges`). Total = total PDF pages; First/Last =
first and last content page; Content = extractable pages
(Last - First + 1). Front and back matter outside this range
will be flagged as `contains_no_content_of_requested_type=true`.

### schweiz

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_schweiz_1994 | 382 | 70 | 337 | 268 |
| mg_schweiz_1995 | 410 | 68 | 354 | 287 |
| mg_schweiz_1996 | 432 | 76 | 374 | 299 |
| mg_schweiz_1998 | 480 | 76 | 413 | 338 |
| mg_schweiz_1999 | 476 | 76 | 381 | 306 |
| mg_schweiz_2000 | 508 | 99 | 445 | 347 |
| mg_schweiz_2001 | 506 | 83 | 405 | 323 |
| mg_schweiz_2002 | 470 | 85 | 409 | 325 |
| mg_schweiz_2003 | 584 | 94 | 509 | 416 |
| mg_schweiz_2004 | 572 | 96 | 498 | 403 |
| mg_schweiz_2005 | 584 | 107 | 519 | 413 |
| mg_schweiz_2006 | 594 | 88 | 498 | 411 |
| mg_schweiz_2007 | 582 | 97 | 508 | 412 |
| mg_schweiz_2011 | 514 | 89 | 452 | 364 |
| mg_schweiz_2013 | 524 | 96 | 466 | 371 |
| mg_schweiz_2014 | 534 | 94 | 466 | 373 |
| mg_schweiz_2016 | 522 | 91 | 466 | 376 |

### benelux

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_benelux_1976 | 356 | 55 | 347 | 293 |
| mg_benelux_1978 | 380 | 55 | 373 | 319 |
| mg_benelux_1979 | 404 | 47 | 392 | 346 |
| mg_benelux_1982 | 420 | 47 | 405 | 359 |
| mg_benelux_1986 | 424 | 47 | 396 | 350 |
| mg_benelux_1988 | 424 | 49 | 411 | 363 |
| mg_benelux_1991 | 468 | 47 | 444 | 398 |
| mg_benelux_1992 | 452 | 47 | 419 | 373 |
| mg_benelux_1993 | 466 | 47 | 431 | 385 |
| mg_benelux_1995 | 512 | 57 | 482 | 426 |
| mg_benelux_1996 | 510 | 61 | 483 | 423 |
| mg_benelux_1998 | 586 | 73 | 539 | 467 |
| mg_benelux_1999 | 608 | 73 | 546 | 474 |
| mg_benelux_2000 | 626 | 88 | 594 | 507 |
| mg_benelux_2001 | 625 | 72 | 572 | 501 |
| mg_benelux_2002 | 604 | 73 | 575 | 503 |
| mg_benelux_2003 | 822 | 73 | 788 | 716 |
| mg_benelux_2004 | 818 | 86 | 800 | 715 |
| mg_benelux_2005 | 834 | 88 | 813 | 726 |
| mg_benelux_2006 | 820 | 74 | 799 | 726 |

### portugal_spain

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_portugal_spain_1975 | 412 | 63 | 405 | 343 |
| mg_portugal_spain_1976 | 420 | 63 | 415 | 353 |
| mg_portugal_spain_1978 | 428 | 63 | 423 | 361 |
| mg_portugal_spain_1979 | 432 | 63 | 427 | 365 |
| mg_portugal_spain_1982 | 476 | 81 | 468 | 388 |
| mg_portugal_spain_1983 | 484 | 81 | 477 | 397 |
| mg_portugal_spain_1984 | 500 | 81 | 497 | 417 |
| mg_portugal_spain_1987 | 530 | 81 | 528 | 448 |
| mg_portugal_spain_1988 | 540 | 79 | 535 | 457 |
| mg_portugal_spain_1989 | 536 | 81 | 530 | 450 |
| mg_portugal_spain_1990 | 544 | 57 | 518 | 462 |
| mg_portugal_spain_1991 | 576 | 63 | 549 | 487 |
| mg_portugal_spain_1992 | 572 | 78 | 555 | 478 |
| mg_portugal_spain_1993 | 596 | 84 | 579 | 496 |
| mg_portugal_spain_1994 | 608 | 84 | 593 | 510 |
| mg_portugal_spain_1995 | 644 | 90 | 617 | 528 |
| mg_portugal_spain_1996 | 668 | 98 | 655 | 558 |
| mg_portugal_spain_1997 | 740 | 98 | 722 | 625 |
| mg_portugal_spain_1998 | 792 | 98 | 768 | 671 |
| mg_portugal_spain_1999 | 824 | 98 | 802 | 705 |
| mg_portugal_spain_2000 | 938 | 136 | 915 | 780 |
| mg_portugal_spain_2003 | 1152 | 116 | 1130 | 1015 |
| mg_portugal_spain_2006 | 1092 | 98 | 1077 | 980 |
| mg_portugal_spain_2007 | 1146 | 118 | 1022 | 905 |
| mg_portugal_spain_2009 | 1300 | 100 | 1218 | 1119 |
| mg_portugal_spain_2010 | 1138 | 70 | 1089 | 1020 |
| mg_portugal_spain_2012 | 1110 | 70 | 1045 | 976 |
| mg_portugal_spain_2015 | 1004 | 86 | 970 | 885 |
| mg_portugal_spain_2016 | 986 | 84 | 936 | 853 |
| mg_portugal_spain_2017 | 908 | 84 | 856 | 773 |
| mg_portugal_spain_2018 | 878 | 84 | 831 | 748 |
| mg_portugal_spain_2019 | 700 | 80 | 690 | 611 |

### britain

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_britain_1976 | 548 | 51 | 519 | 469 |
| mg_britain_1979 | 608 | 61 | 591 | 531 |
| mg_britain_1980 | 616 | 61 | 585 | 525 |
| mg_britain_1982 | 612 | 61 | 576 | 516 |
| mg_britain_1985 | 616 | 61 | 600 | 540 |
| mg_britain_1986 | 636 | 61 | 605 | 545 |
| mg_britain_1988 | 682 | 61 | 667 | 607 |
| mg_britain_1989 | 680 | 63 | 667 | 605 |
| mg_britain_1990 | 708 | 57 | 678 | 622 |
| mg_britain_1991 | 734 | 57 | 701 | 645 |
| mg_britain_1992 | 720 | 57 | 697 | 641 |
| mg_britain_1993 | 748 | 59 | 712 | 654 |
| mg_britain_1994 | 758 | 57 | 727 | 671 |
| mg_britain_1995 | 748 | 61 | 730 | 670 |
| mg_britain_1996 | 767 | 71 | 751 | 681 |
| mg_britain_1997 | 848 | 80 | 833 | 754 |
| mg_britain_1998 | 906 | 78 | 874 | 797 |
| mg_britain_1999 | 926 | 82 | 911 | 830 |
| mg_britain_2000 | 972 | 104 | 946 | 843 |
| mg_britain_2002 | 988 | 90 | 946 | 857 |
| mg_britain_2003 | 1156 | 92 | 1124 | 1033 |
| mg_britain_2004 | 1156 | 106 | 1128 | 1023 |
| mg_britain_2005 | 1160 | 104 | 1114 | 1011 |
| mg_britain_2006 | 1108 | 79 | 1081 | 1003 |
| mg_britain_2007 | 1092 | 86 | 1053 | 968 |
| mg_britain_2008 | 1288 | 72 | 1181 | 1110 |
| mg_britain_2009 | 1284 | 74 | 1157 | 1084 |
| mg_britain_2010 | 1092 | 50 | 994 | 945 |
| mg_britain_2011 | 1002 | 53 | 932 | 880 |
| mg_britain_2012 | 1042 | 51 | 967 | 917 |
| mg_britain_2013 | 1042 | 51 | 961 | 911 |
| mg_britain_2014 | 1042 | 55 | 969 | 915 |
| mg_britain_2015 | 1084 | 82 | 1037 | 956 |
| mg_britain_2016 | 1162 | 50 | 1121 | 1072 |
| mg_britain_2017 | 947 | 54 | 910 | 857 |
| mg_britain_2018 | 914 | 50 | 859 | 810 |
| mg_britain_2019 | 760 | 44 | 725 | 682 |

### deutschland

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_deutschland_1977 | 800 | 61 | 785 | 725 |
| mg_deutschland_1978 | 788 | 61 | 778 | 718 |
| mg_deutschland_1986 | 880 | 61 | 869 | 809 |
| mg_deutschland_1987 | 892 | 61 | 880 | 820 |
| mg_deutschland_1989 | 912 | 61 | 893 | 833 |
| mg_deutschland_1990 | 940 | 52 | 904 | 853 |
| mg_deutschland_1991 | 952 | 58 | 914 | 857 |
| mg_deutschland_1992 | 940 | 58 | 918 | 861 |
| mg_deutschland_1993 | 940 | 58 | 926 | 869 |
| mg_deutschland_1994 | 988 | 58 | 975 | 918 |
| mg_deutschland_1995 | 1028 | 64 | 1009 | 946 |
| mg_deutschland_1996 | 1052 | 66 | 1028 | 963 |
| mg_deutschland_1997 | 1138 | 66 | 1113 | 1048 |
| mg_deutschland_1998 | 1204 | 74 | 1174 | 1101 |
| mg_deutschland_1999 | 1232 | 74 | 1196 | 1123 |
| mg_deutschland_2000 | 1260 | 91 | 1246 | 1156 |
| mg_deutschland_2001 | 1260 | 75 | 1243 | 1169 |
| mg_deutschland_2002 | 1246 | 82 | 1226 | 1145 |
| mg_deutschland_2003 | 1628 | 94 | 1602 | 1509 |
| mg_deutschland_2004 | 1562 | 97 | 1547 | 1451 |
| mg_deutschland_2005 | 1490 | 102 | 1483 | 1382 |
| mg_deutschland_2006 | 1692 | 76 | 1609 | 1534 |
| mg_deutschland_2007 | 1648 | 85 | 1553 | 1469 |
| mg_deutschland_2008 | 1530 | 88 | 1431 | 1344 |
| mg_deutschland_2009 | 1524 | 85 | 1443 | 1359 |
| mg_deutschland_2010 | 1458 | 64 | 1361 | 1298 |
| mg_deutschland_2011 | 1440 | 61 | 1355 | 1295 |
| mg_deutschland_2013 | 1494 | 72 | 1410 | 1339 |
| mg_deutschland_2014 | 1408 | 72 | 1325 | 1254 |
| mg_deutschland_2015 | 1420 | 118 | 1382 | 1265 |
| mg_deutschland_2016 | 1344 | 116 | 1291 | 1176 |
| mg_deutschland_2017 | 1194 | 114 | 1139 | 1026 |
| mg_deutschland_2018 | 1126 | 114 | 1079 | 966 |

### italy

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_italy_1975 | 692 | 87 | 679 | 593 |
| mg_italy_1977 | 684 | 59 | 676 | 618 |
| mg_italy_1979 | 684 | 59 | 676 | 618 |
| mg_italy_1980 | 692 | 59 | 685 | 627 |
| mg_italy_1983 | 680 | 59 | 659 | 601 |
| mg_italy_1984 | 680 | 59 | 667 | 609 |
| mg_italy_1985 | 696 | 59 | 689 | 631 |
| mg_italy_1988 | 720 | 59 | 713 | 655 |
| mg_italy_1989 | 720 | 59 | 712 | 654 |
| mg_italy_1990 | 722 | 56 | 707 | 652 |
| mg_italy_1991 | 740 | 56 | 721 | 666 |
| mg_italy_1992 | 748 | 56 | 729 | 674 |
| mg_italy_1993 | 764 | 56 | 742 | 687 |
| mg_italy_1994 | 756 | 64 | 738 | 675 |
| mg_italy_1995 | 776 | 64 | 761 | 698 |
| mg_italy_1996 | 812 | 66 | 788 | 723 |
| mg_italy_1997 | 882 | 68 | 867 | 800 |
| mg_italy_1998 | 912 | 68 | 897 | 830 |
| mg_italy_1999 | 932 | 68 | 899 | 832 |
| mg_italy_2000 | 938 | 83 | 918 | 836 |
| mg_italy_2001 | 934 | 65 | 909 | 845 |
| mg_italy_2002 | 950 | 73 | 929 | 857 |
| mg_italy_2003 | 1196 | 72 | 1171 | 1100 |
| mg_italy_2004 | 1190 | 87 | 1174 | 1088 |
| mg_italy_2005 | 1186 | 87 | 1170 | 1084 |
| mg_italy_2006 | 1134 | 95 | 1118 | 1024 |
| mg_italy_2007 | 1426 | 76 | 1358 | 1283 |
| mg_italy_2008 | 1388 | 80 | 1324 | 1245 |
| mg_italy_2011 | 1402 | 67 | 1305 | 1239 |
| mg_italy_2012 | 1416 | 72 | 1319 | 1248 |
| mg_italy_2013 | 1372 | 74 | 1306 | 1233 |
| mg_italy_2014 | 1278 | 72 | 1217 | 1146 |
| mg_italy_2016 | 1372 | 92 | 1324 | 1233 |
| mg_italy_2017 | 1352 | 84 | 1271 | 1188 |
| mg_italy_2018 | 1328 | 88 | 1263 | 1176 |
| mg_italy_2019 | 1110 | 80 | 1043 | 964 |

### france

| Edition | Total | First | Last | Content |
|---|---|---|---|---|
| mg_france_1975 | 1196 | 69 | 1184 | 1116 |
| mg_france_1976 | 1196 | 71 | 1189 | 1119 |
| mg_france_1977 | 1208 | 71 | 1197 | 1127 |
| mg_france_1978 | 1216 | 71 | 1205 | 1135 |
| mg_france_1979 | 1222 | 71 | 1210 | 1140 |
| mg_france_1980 | 1236 | 71 | 1213 | 1143 |
| mg_france_1981 | 1238 | 71 | 1225 | 1155 |
| mg_france_1982 | 1266 | 71 | 1225 | 1155 |
| mg_france_1983 | 1268 | 71 | 1230 | 1160 |
| mg_france_1984 | 1256 | 71 | 1239 | 1169 |
| mg_france_1985 | 1297 | 81 | 1257 | 1177 |
| mg_france_1986 | 1301 | 81 | 1268 | 1188 |
| mg_france_1987 | 1300 | 81 | 1270 | 1190 |
| mg_france_1988 | 1322 | 81 | 1291 | 1211 |
| mg_france_1989 | 1224 | 42 | 1211 | 1170 |
| mg_france_1990 | 1252 | 46 | 1234 | 1189 |
| mg_france_1991 | 1296 | 46 | 1280 | 1235 |
| mg_france_1992 | 1318 | 46 | 1305 | 1260 |
| mg_france_1993 | 1336 | 46 | 1316 | 1271 |
| mg_france_1994 | 1352 | 46 | 1299 | 1254 |
| mg_france_1995 | 1352 | 46 | 1337 | 1292 |
| mg_france_1996 | 1354 | 48 | 1329 | 1282 |
| mg_france_1997 | 1492 | 48 | 1459 | 1412 |
| mg_france_1998 | 1500 | 48 | 1477 | 1430 |
| mg_france_1999 | 1524 | 48 | 1492 | 1445 |
| mg_france_2000 | 1504 | 128 | 1467 | 1340 |
| mg_france_2001 | 1768 | 55 | 1703 | 1649 |
| mg_france_2002 | 1766 | 69 | 1745 | 1677 |
| mg_france_2003 | 1802 | 117 | 1777 | 1661 |
| mg_france_2004 | 1826 | 121 | 1798 | 1678 |
| mg_france_2005 | 1880 | 125 | 1864 | 1740 |
| mg_france_2006 | 2130 | 103 | 2067 | 1965 |
| mg_france_2007 | 2128 | 101 | 2057 | 1957 |
| mg_france_2008 | 2076 | 102 | 2011 | 1910 |
| mg_france_2009 | 2126 | 103 | 2060 | 1958 |
| mg_france_2010 | 1934 | 82 | 1863 | 1782 |
| mg_france_2011 | 1932 | 90 | 1862 | 1773 |
| mg_france_2012 | 2030 | 86 | 1963 | 1878 |
| mg_france_2013 | 2024 | 86 | 1954 | 1869 |
| mg_france_2014 | 2032 | 80 | 1967 | 1888 |
| mg_france_2015 | 1938 | 84 | 1901 | 1818 |
| mg_france_2016 | 2118 | 88 | 2044 | 1957 |
| mg_france_2017 | 2018 | 84 | 1937 | 1854 |
| mg_france_2018 | 1896 | 90 | 1799 | 1710 |
| mg_france_2019 | 1528 | 90 | 1448 | 1359 |
| mg_france_2020 | 1340 | 90 | 1260 | 1171 |
| mg_france_2021 | 1348 | 64 | 1264 | 1201 |
| mg_france_2022 | 1132 | 78 | 1073 | 996 |
| mg_france_2023 | 1266 | 135 | 1163 | 1029 |
