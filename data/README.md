# Data

The source is the **100 Portfolios Formed on Size and Book-to-Market
(10 x 10), Daily** dataset from
[Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

`data/raw/` is intentionally excluded from version control. Download the
official CSV ZIP into that directory and use `A_Data/data_preprocessing.py` to
regenerate the processed files.

`data/processed/returns_equal_weighted_98.csv` is stored once for convenient
reproduction of the model results. Values are percentage points. The
processed universe contains 98 assets after removing `ME9 BM10` and
`BIG HiBM`.
