### Pip Requirements
pip install numpy

## Usage
```bash
python .\main.py <data file> <population size> <number of generations> <selection method> <selection rate> <crossover method> <crossover rate> <mutation rate>
```
#### Note
For Uniform crossover, please still enter a crossover rate, but it will be ignored.


## Example Usages
##### Elitist 
python .\main.py .\genAlgData1.txt 150 10 Elitist 90 K-point 1 0.50

##### Roulette
python .\main.py .\debug.txt 150 10 Roulette 20 K-point 1 0.50

##### Uniform
 python .\main.py .\debug.txt 150 10 Roulette 20 Uniform 1 0.50

##### K-point
python .\main.py .\debug.txt 150 10 Roulette 20 K-point 1 0.50

### Best Performances I've seen
##### python .\main.py .\genAlgData1.txt 150 10 Elitist 99 Uniform 1 0.05
##### fitness: 3900

##### python .\main.py .\genAlgData1.txt 200 50 Elitist 20 Uniform 1 0.01
##### fitness: 4902


##### python .\main.py .\genAlgData2.txt 150 100 Elitist 30 K-point 1 0.01
##### fitness: 400


##### python .\main.py .\genAlgData2.txt 200 50 Elitist 30 Uniform 1 0.01
##### fitness: 420