# GRAB: A Challenging <ins>GR</ins>aph <ins>A</ins>nalysis <ins>B</ins>enchmark for Large Multimodal Models

### [[Project Page](https://grab-benchmark.github.io)] [[Paper]()] [[Data](https://huggingface.co/datasets/jonathan-roberts1/GRAB)] [[Code](https://github.com/jonathan-roberts1/GRAB/)] [[Leaderboard](https://grab-benchmark.github.io)]

### Jonathan Roberts, Kai Han, Samuel Albanie

Overview of GRAB.

This accompanying repository contains...

### Contents
  - [News](#news)
  - [Key Insights](#key-insights)
  - [Data](#data)
  - [Evaluation](#evaluation)

    
## News
🎉 **[22/08/24]** Initial release!

## Key Insights

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

### Gemini using downloaded images file
TODO

### Automatic evaluation
TODO









