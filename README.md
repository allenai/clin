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

Example command to run ChatGPT baseline for ScienceWorld
```
python scienceworld/chatgpt_agent.py --task-num "1" --env-step-limit 3 --var-num 1 --num-episodes 1 --gpt-model "gpt-4" --output-path-prefix logs/temp
```
