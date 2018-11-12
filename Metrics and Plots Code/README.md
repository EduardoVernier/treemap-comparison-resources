# Caching metrics
for d in $(cat tvcg-datasets.txt); do echo $d; time python3 Main.py cache-metrics $d; done;

# Star scatter
python3 Main.py scatter $(cat tvcg-datasets.txt)

# Matrices
python3 Main.py matrix $(cat tvcg-datasets.txt)

# Boxplots
for d in $(cat tvcg-datasets.txt); do echo $d; time python3 Main.py boxplots $d; done

# KDE
for d in $(cat tvcg-datasets.txt); do echo $d; time python3 Main.py kde-ct $d; time python3 Main.py kde-rpc $d; done;