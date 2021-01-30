for beta in 0.5 0.3 0.1 0.7 0.9
do
  python trainfull.py -m "beta=$beta" -b $beta -d bbn
done