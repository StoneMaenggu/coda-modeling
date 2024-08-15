
#!/bin/bash
cd ~/HDD1/dataset/aihub_sign
export AIHUB_ID=ox4336@pusan.ac.kr
export AIHUB_PW='oxJ52797!'
# mkdir port1
# cd port1

for i in 39477 39478 39479 494853
do
    aihubshell -mode d -datasetkey 103 -filekey ${i}
done

# aihubshell -mode d -datasetkey 103 -filekey 39602
