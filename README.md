# Essay_HAN
Automated essay grading with Hierarchical Attention Networks

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

I'm assuming you're using anaconda distribution and 3.5+ version of python.

Clone repository

```
git clone https://github.com/rikayi/Essay_HAN.git
cd DRU-DL-Project-Structure
```

Install requirements

```
pip install -r requirements.txt
```

Open data/preprocessing.ipynb and run notebook. If you wish you can play around with preprocessing parameters.

After preprocessing you are ready to train model.

```
cd mains
python main.py -c ../configs/config.json
```
You should see something like this in your terminal 
![alt text](https://github.com/rikayi/Essay_HAN/blob/master/resources/prog_bar.PNG)

Open one more terminal window and run tensorboard. Watch metrics in near real-time
![alt text](https://github.com/rikayi/Essay_HAN/blob/master/resources/tb.PNG)


## Acknowledgments

* [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) 
* [ASAP-dataset](https://github.com/GregoryZeng/ASAP-dataset)
* [Parts of model implimentation](https://github.com/ematvey)
