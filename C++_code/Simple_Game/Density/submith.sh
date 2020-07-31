#!/bin/bash

for i in 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90
do
	NAME='DensityRHO'${i}'.dat'
	./density ${NAME} 500 ${i}
done