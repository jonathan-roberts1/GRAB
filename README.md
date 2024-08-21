# GRAB: A Challenging <ins>GR</ins>aph <ins>A</ins>nalysis <ins>B</ins>enchmark for Large Multimodal Models

### [[Project Page](https://grab-benchmark.github.io)] [[Paper]()] [[Data](https://huggingface.co/datasets/jonathan-roberts1/GRAB)] [[Code](https://github.com/jonathan-roberts1/GRAB/)] [[Leaderboard](https://grab-benchmark.github.io)]

### Jonathan Roberts, Kai Han, Samuel Albanie

Large multimodal models (LMMs) have exhibited proficiencies across many visual tasks. Although numerous benchmarks exist to evaluate model performance, they increasingly have insufficient headroom and are **unfit to evaluate the next generation of frontier LMMs**.

To overcome this, we present **GRAB**, a challenging benchmark focused on the tasks **human analysts** might typically perform when interpreting figures. Such tasks include estimating the mean, intercepts or correlations of functions and data series and performing transforms.

We evaluate a suite of **20 LMMs** on GRAB, finding it to be a challenging benchmark, with the current best model scoring just **21.7%**.

#### This repository contains a summary of the key findings from our paper, instructions and example code to both access the GRAB benchmark data and evaluate on it.
### Contents
  - [News](#news)
  - [Key Information](#key-information)
  - [Data](#data)
  - [Evaluation](#evaluation)

## News
🎉 **[22/08/24]** Initial release!

## Key Information
- GRAB contains 2170 graph analysis questions
- Each question includes a unique synthetically generated graph
- The questions test graph analysis capabilities of varying complexity
- Some questions include multiple series, function or transforms
- GRAB proves challenging for current models, with the highest performing scoring just 21.7%.

## Data

GRAB data can be accessed either using the HuggingFace datasets library or by manual download.

### Option 1: HuggingFace datasets

```python
from datasets import load_dataset

# load dataset
grab_dataset = load_dataset("jonathan-roberts1/GRAB", split='GRAB')

"""
Dataset({
    features: ['pid', 'question', 'decoded_image', 'image', 'answer', 'task', 'category', 'complexity'],
    num_rows: 2170
})
"""

# query individual questions
grab_dataset[40] # e.g., the 41st element
"""
{'pid': 40, 'question': 'What is the value of the y-intercept of the function? Give your answer as an integer.',
'decoded_image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=5836x4842 at 0x12288EA60>,
'image': 'images/40.png', 'answer': '1', 'task': 'properties', 'category': 'Intercepts and Gradients',
'complexity': 0}
"""
```

### Option 2: Manual download

Directly downloading image files and question data from the GRAB HuggingFace repository into the ```data``` directory in this repo.
```
cd data
wget https://huggingface.co/datasets/jonathan-roberts1/GRAB/resolve/main/images.zip
unzip images.zip && rm images.zip
```
#### Expected structure
```
├── data
    ├── grab.json
    ├── images
        ├── 1.png
        ├── 2.png
        └── ...
```

Note: ```images/``` needs to be downloaded.

### Option 3: Hybrid
For some applications, a hybrid option that, for example, uses the manually downloaded image files and HF datasets for question data. This approach can make use of the 'image' feature of each item in the Dataset, which includes filenames in the ```images``` dir.

## Evaluation

### Example inference and evaluation using LLaVA-1.5 7b
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
import torch
from tqdm import tqdm
import pandas as pd

# load grab
grab_dataset = load_dataset("jonathan-roberts1/GRAB", split="GRAB")
# optional: set cache_dir="PATH/TO/MY/CACHE/DIR"

model_name = "llava-hf/llava-1.5-7b-hf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataframe to store results
output_df = pd.DataFrame(columns=["Question_ID", "Model Output", 
                                  "Model Answer", "Ground Truth", "Correct?"])

# initialise model and processor
model = LlavaForConditionalGeneration.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# example prompt template
prompt_template = \
"""
{question}\n Only provide the answer, no reasoning steps.
If you are unsure, still provide an answer. Answer: \n
"""

# Iterate over questions
for item in tqdm(grab_dataset):

    question = item['question'] # with precision instructions
    image = item['decoded_image'] # PIL image
    true_answer = item['answer'] # ground truth answer

    # construct simple prompt
    prompt = prompt_template.format(question=question)

    # inference
    conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                        ],
                    },
                    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(formatted_prompt, image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    output = processor.decode(output[0], skip_special_tokens=True)
    output = output.split('ASSISTANT: ')[1].lstrip() # extract response
    model_answer = output[:len(true_answer)] # extract model answer

    # exact matching eval
    answer_eval = True if model_answer == true_answer else False

    results_row = {"Question_ID": item['pid'], "Model Output": output,
                    "Model Answer": model_answer, "Ground Truth": true_answer, 
                    "Correct?": answer_eval}
    output_df = pd.concat([output_df, pd.DataFrame([results_row])], ignore_index=True)

# save output
#output_df.to_csv("PATH/TO/SAVE/DIR", index=False)

# compute accuracy
grab_accuracy = output_df["Correct?"].mean()
print(f"GRAB Accuracy: {100 * grab_accuracy:.2f}%")
```

### Inference using ```images``` and ```grab.json```

```python
import json
from PIL import Image


with open('data/grab.json', 'r') as f:
    grab_dataset_dict = json.load(f)

for _, value in grab_dataset_dict.items():
  question = value['question']
  image_filepath = 'data/' + value['image']
  img = Image.open(image_filepath)

  # construct prompt using question and img...
```

### Evaluation with LLM
Especially for LMMs that tend to generate verbose output, performance on GRAB can be increased by leveraging an LLM to automatically extract the answer.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch


def llama3_answer_extraction(question, model_output, tokenizer, model):

    prompt_template = \
    """    
    A generative model has been asked this question: "{question}" about a plot.\n
    The output from the model answering the question is: "{output}"\n
    Extract just the answer from the generative model output. Maintain the same precision given by the model. 
    Convert any numbers to digits (e.g., "one" to "1"). Remove any additional terms like 'approximately'.
    Return only the extracted numeric answer, without any additional text or explanation. If no answer is provided, return "None".
    """
    prompt = prompt_template.format(question=question, output=model_output)
    
    messages = [
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(model.device)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=False,
        temperature=None,
        top_p=None
    )
    response = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    print(response_text)
    return response_text

# -- example usage -- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load open source model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
                                        'meta-llama/Meta-Llama-3.1-8B-Instruct',
                                        torch_dtype=torch.bfloat16,
                                        ).to(device)
question: str 
model_output: str # generated previously
extracted_answer = llama3_answer_extraction(question, model_output, tokenizer, model)         

```






