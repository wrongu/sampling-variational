#!/bin/bash
  
DISTRIBS=(banana cigar laplace)
COMPONENTS=(MVNIso)
SAMPLES=10000
# essentially logspace(0,2,10)
LAMBDAS=(1.0 1.5848931924611136 2.51188643150958 3.9810717055349722 6.309573444801933 10.0 15.848931924611133 25.118864315095795 39.810717055349734 63.09573444801933 100.0)
CHAINS=(0)

parallel $1 --tag --line-buffer --jobs=30 python -m scripts.run_hmc --distrib={1} --component={2} --lam={3} --save-dir=notebooks/results/ --samples=$SAMPLES --chain={4} \
 ::: ${DISTRIBS[@]} ::: ${COMPONENTS[@]} ::: ${LAMBDAS[@]} ::: ${CHAINS[@]}

