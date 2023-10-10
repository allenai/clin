# CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization

This directory contains code to do interactive continual learning with ScienceWorld environment.

Install Java
You will have to have Java 1.8+ installed on your system (shipped with most linux distributions).

Create a new Python environment
```
conda create --name sw python=3.8
conda activate sw
pip install -r requirements.txt

export PYTHONPATH=.         # prevents any module errors
export OPENAI_API_KEY=<your key>  # would need access to one of these 2 models: gpt-3.5-turbo, gpt-4

```

Alinging to the latest SW; somewhere outside the CLIN repo

```
cd ..
git clone https://github.com/allenai/ScienceWorld.git
cd ScienceWorld
git checkout exhaustivevalidactions
git pull
pip install -e .
```

Get back into CLIN repo

```
cd ../clin
mkdir logs
```

Example command to run CLIN agent for ScienceWorld
```
python scienceworld/clin_agent.py --task-num "4" --var-num 1 --env-step-limit 2 --num-episodes 1  --gpt-model "gpt-4-0613" --summarize_end_of_episode 1  --device "cpu"  --temperature 0.0  --use-gold-memory-in-ep0 0 --gold-traces "" --use-last-k-memories 0 --quadrant 1 --simplifications-preset "easy" --output-path-prefix logs/testrun/
```
