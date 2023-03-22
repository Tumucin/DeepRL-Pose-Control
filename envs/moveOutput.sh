#!/bin/bash

for i in {70..101}
do
	mv /kuacc/users/tbal21/.conda/envs/stableBaselines/panda-gym/panda_gym/envs/PPOexp${i}.out /kuacc/users/tbal21/.conda/envs/stableBaselines/panda-gym/panda_gym/envs/shOutput/ 
done
