import torch
import common
from transformers import AutoModelForCausalLM
from torch.nn.functional import softmax
import time



def greedy_sampling(next_token_logits, tokenizer, generated):
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated = torch.cat((generated, next_token_id.unsqueeze(-1)), dim=1)
    next_token_text = tokenizer.decode(next_token_id)

    return generated, next_token_id, next_token_text

def top_p_sampling(next_token_logits, tokenizer, top_p, generated):
    # Apply softmax to get probabilities
    probabilities = softmax(next_token_logits, dim=-1)

    # Sort tokens by probability
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep only tokens within the top-p threshold
    top_p_mask = cumulative_probs <= top_p
    top_p_mask[..., 0] = True  # Ensure at least one token is always included

    # Re-normalize the probabilities of the selected tokens
    filtered_probs = sorted_probs * top_p_mask
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Avoid division by zero

    # Sample from the filtered distribution
    next_token_id = sorted_indices.gather(1, torch.multinomial(filtered_probs, num_samples=1))

    # Ensure shape is [1, 1]
    next_token_id = next_token_id.squeeze(-1).unsqueeze(1)

    # Concatenate to generated sequence
    generated = torch.cat((generated, next_token_id), dim=1)

    next_token_text = tokenizer.decode(next_token_id.squeeze().tolist())    

    return generated, next_token_id, next_token_text

# Test Model
def generate(method, model, tokenizer, ori_prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p, show_debug: bool = True):
    model.eval()
    if method in ['coreinfer', 'dense', 'partinfer']:
        prompt = process_prompt_stable(ori_prompt, task_type, num_fewshot)
        
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_token_length = input_ids.shape[-1]
    if show_debug:
        print(f"Prompt token length: {prompt_token_length}")
    pre_fill_start_time = time.time()
    if show_debug:
        print("Starting the prefilling stage...", end="")
    
    eos_token_id = tokenizer.convert_tokens_to_ids('.')
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    pre_fill_end_time = time.time()
    pre_fill_elapsed_time = pre_fill_end_time - pre_fill_start_time
    if show_debug:
        print(f"Done. Prefilling stage calculated in {pre_fill_elapsed_time:.2f} seconds.\n")

    generated = input_ids
    
    print(ori_prompt) 
    start_time = time.time()
    counter = 0
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
    
        next_token_logits = logits[:, -1, :]

        if sampling_method == "greedy":
            generated, next_token_id, next_token_text = greedy_sampling(next_token_logits, tokenizer, generated)
        elif sampling_method == "top-p":
            generated, next_token_id, next_token_text = top_p_sampling(next_token_logits, tokenizer, top_p, generated)
        
        print(next_token_text, end='', flush=True)
        if next_token_id.item() == eos_token_id:
            break
        if '.' in next_token_text:
            break
        counter += 1
    end_time = time.time()
    
    num_generated_tokens = generated.size(1) - input_ids.size(1)
    
    elapsed_time = end_time - start_time
    tokens_per_second = num_generated_tokens / elapsed_time
    
    if show_debug:
        print(f'\n\nGenerated {num_generated_tokens} tokens in {elapsed_time:.2f} seconds.')
        print(f'Decoding speed: {tokens_per_second:.2f} tokens/second')


def process_prompt_stable(prompt, task_type, num_fewshot):
    if num_fewshot == 0:
        return prompt
    
    if task_type == 'QA':
        pre_prompt = generate_few_shot_QA(num_fewshot)
        final_prompt = pre_prompt + prompt + '\nA:'
        
    elif task_type == 'Summarize':
        # pre_prompt = 'Summarize the following document: '
        pre_prompt = generate_few_shot_summarize(num_fewshot)
        final_prompt = pre_prompt + prompt
        # final_prompt = prompt
        
    elif task_type == 'translate_de_en':
        pre_prompt = generate_few_shot_translate_de_en(num_fewshot)
        final_prompt = pre_prompt + prompt + '\nEnglish phrase:'

    else:
        print("Task_type must be one of QA, Summarize and translate_de_en")
    
    return final_prompt

    



def generate_few_shot_QA(num_fewshot):
    # All available few-shot examples
    examples = [
        "Q: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.",
        "Q: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.",
        "Q: Which party did he belong to?\nA: He belonged to the Republican Party.",
        "Q: What is the square root of banana?\nA: I have no comment.",
        "Q: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.",
        "Q: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain."
    ]
    
    if num_fewshot < 1 or num_fewshot > len(examples):
        raise ValueError("Number of few-shot examples must be between 1 and 6.")
    
    few_shot_prompt = '\n\n'.join(examples[:num_fewshot])
    pre_prompt = few_shot_prompt + '\n\nQ: '
    
    return pre_prompt





def generate_few_shot_translate_de_en(num_fewshot):
    # All available few-shot examples
    examples = [
    "German phrase: Seiner Ansicht nach könnten alle Mitglieder beider Vereine künftig wieder an einem Strang ziehen.\nEnglish phrase: In his opinion, all members of each club could come together again in the future.",
    "German phrase: In Bad Salzhausen wurden die Sportler von Bürgermeister Hans-Peter Seum, der in Bad Vilbel mit dem Fahrrad mitgestartet war, und der Leiterin des Eigenbetriebs Bad Salzhausen, Petra Schwing-Döring, begrüßt.\nEnglish phrase: In Bad Salzhausen, the sportsmen-and-women were greeted by mayor Hans-Peter Seum, who had ridden off the start-line in Bad Vilbel on his own bike, and the boss of the municipal company Bad Salzhausen, Petra Schwing-Döring.",
    "German phrase: Als die Vereinigten Staaten nach dem Zweiten Weltkrieg Japan besetzte, ermutigten General Douglas Mac Arthur und sein Stab das Land dazu, eine Verfassung zu verabschieden, die sicherstellen solle, dass die militarisierte Autokratie Hedki Tojos durch Demokratie ersetzt würde.\nEnglish phrase: When the United States occupied Japan after World War II, General Douglas MacArthur and his aides encouraged the country to adopt a constitution designed to assure that Hideki Tojo's militarized autocracy would be replaced with democracy.",
    "German phrase: Henry führte 20 Zeilen aus Othellos letzter Rede für den Dokumentarfilm auf und er war begeistert.\nEnglish phrase: Henry delivered 20 lines of Othello's last speech for the documentary and he was hooked.",
    "German phrase: Für Adelaide ist am Dienstag eine Höchsttemperatur von 16°C vorhergesagt, mit möglichen Schauern.\nEnglish phrase: A top of 16C is forecast for Adelaide on Tuesday, with the chance of a shower or two.",
    "German phrase: Und es ist anstrengend.\nEnglish phrase: And it's tedious."]
    
    if num_fewshot < 1 or num_fewshot > len(examples):
        raise ValueError("Number of few-shot examples must be between 1 and 6.")
    
    few_shot_prompt = '\n\n'.join(examples[:num_fewshot])
    pre_prompt = few_shot_prompt + 'German phrase: '
    
    return pre_prompt






def generate_few_shot_summarize(num_fewshot):
    # All available few-shot examples
    examples = [
    "Text: The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail. 'It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten,' she said. 'That may not be true but it is perhaps my perspective over the last few days. 'Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?' Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process. 'I was quite taken aback by the amount of damage that has been done,' he said. 'Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses.' He said it was important that 'immediate steps' were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.\nSummary: Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
    
    "Text: he announcement ends months of uncertainty for Cornish Language Partnership staff whose contracts had been due to end. Local government minister Andrew Stunnell said the three-year funding package for the service would help make sure the language survived. But he warned that long term funding should come from Cornwall. He said it was 'important to make sure the Cornish were given the opportunity to put down sound foundations.' 'In the longer term support for the Cornish language is going to be something which is going to have to be based in Cornwall and will not come from London,' he added. The Cornish Language Partnership's, Jennifer Lowe, said: 'We can now plan for the future thanks to the funding.' The United Nations recently upgraded the status of the Cornish language from 'extinct' to 'critically endangered'. It is thought fewer than 500 people worldwide are fluent in the language.\nSummary: The government is spending nearly £400,000 to help save the Cornish language.",

    "Text: Operation Equinox is investigating claims of sexual, physical and emotional abuse between the 1940s and 1990s. In a letter to victims Nottinghamshire Police confirmed 530 of 636 reported crimes were on council property. Officers also said 485 alleged offences were committed by council staff and of 432 suspects, 283 had been identified. More on this story and other news in Nottinghamshire So far, police have had 290 people report crimes. Operation Equinox combined two police inquiries. Operation Daybreak, sent up in 2011, was focussed on the Beechwood children's home in Nottingham, while Operation Xeres has been looking at residential homes in the county. The letter emphasises the progress already made, with former social worker Andris Logins jailed for 20 years. Two other men have been jailed for historical attacks not connected to children's homes and three more trials are due to begin in early 2017. Nottinghamshire Police has not commented directly as the information is part of an ongoing enquiry.\nSummary: An inquiry into historical abuse in Nottinghamshire has recorded more than 500 offences on council property."]
    
    if num_fewshot < 1 or num_fewshot > len(examples):
        raise ValueError("Number of few-shot examples must be between 1 and 3.")
    
    few_shot_prompt = '\n\n'.join(examples[:num_fewshot])
    pre_prompt = few_shot_prompt + '\nSummary: '
    
    return pre_prompt



def process_data(dataset, dataset_name):
    
    if dataset_name == "truthful_qa":
        question = dataset['validation']['question']
        best_answer = dataset['validation']['best_answer']
        process_data = ['Question:' + a + 'Answer:' + b for a, b in zip(question, best_answer)]

    elif dataset_name == "trivia_qa":
        question = dataset['validation']['question']
        answer = [x["value"] for x in dataset["validation"]["answer"]]
        process_data = ['Question: ' + a + 'Answer: ' + b for a, b in zip(question, answer)]
        
    elif dataset_name == "wmt16-de-en":
        de = [x["de"] for x in dataset["validation"]["translation"]]
        en = [x["en"] for x in dataset["validation"]["translation"]]
        process_data = ['German phrase: ' + a + "\nEnglish phrase: " + b for a, b in zip(de, en)]
        
    elif dataset_name == "squadv2":
        filtered_dataset = dataset["validation"].filter(lambda x: len(x["answers"]["text"]) > 0)
        question = filtered_dataset["question"]
        answer = [x["text"][0] for x in filtered_dataset["answers"]]
        context = filtered_dataset["context"]
        process_data = ['Context: ' + a + '\nQuestion: ' + b + '\nAnswer: ' + c for a, b, c in zip(context, question, answer)]
        
    elif dataset_name == "commonsense_qa":
        process_data = []
        for i in range(len(dataset["validation"])):
            item = dataset["validation"][i]
            question = item["question"]
            choices = item["choices"]

            answer_to_index = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3,
                "E": 4
            }

            choices_A = choices["text"][0]
            choices_B = choices["text"][1]
            choices_C = choices["text"][2]
            choices_D = choices["text"][3]
            choices_E = choices["text"][4]

            answer = choices["text"][answer_to_index[item["answerKey"]]]
            
            process_data.append(f"Question: {question}\nA. {choices_A}\nB. {choices_B}\nC. {choices_C}\nD. {choices_D}\nE. {choices_E}\nAnswer: {answer}")
            
    elif dataset_name == "bertaqa_en":
        question = dataset["test"]["question"]
        candidates = dataset["test"]["candidates"]
        answers = [candidates[i][x] for i,x in enumerate(dataset["test"]["answer"])]
        candidates_text = [f"A: {x[0]}\nB: {x[1]}\nC: {x[2]}" for x in dataset["test"]["candidates"]]
        process_data = ['Question: ' + a + '\n' + b + '\nAnswer: ' + c for a, b, c in zip(question, candidates_text, answers)]
        
    elif dataset_name == "lambada":
        texts = dataset["validation"]["text"]
        process_data = ['Context: ' +' '.join(x.split(' ')[:-1]) + '\nPredicted word: ' + x.split(' ')[-1] for x in texts]
        
    elif dataset_name == "careqa_open":
        question = dataset['test']['question']
        answer = dataset['test']['answer']
        process_data = ['Question: ' + a + '\nAnswer: ' + b for a, b in zip(question, answer)]
        
    elif dataset_name == "medtext":
        question = dataset['train']['Prompt']
        answer = dataset['train']['Completion']
        process_data = ['Question: ' + a + '\nAnswer: ' + b for a, b in zip(question, answer)]
        
    elif dataset_name == "meqsum":
        text = dataset['train']['CHQ']
        summary = dataset['train']['Summary']
        process_data = ['Text: ' + a + '\nSummary: ' + b for a, b in zip(text, summary)]
            
    return process_data



def read_indecies_file(file_path, model_name, start_num, end_num):
    num_neurons = common.MODEL_INFO[model_name]["num_neurons"]

    results = []
    current_results = None
    current_layers_neurons = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Applying sparsity"):
                
                if current_results is not None:
                    results.append(current_results)
                
                task_name = line.split(" ")[-1]
                sparsity_name = line.split(" ")[2]
                current_results = {
                    "task_name": task_name,
                    "sparsity_name": sparsity_name,
                    "results": []
                }
            elif line.startswith("Layer"):
                neurons_num = int(line.split(" ")[-1])
                layer_num = int(line.split(" ")[1].split(":")[0])
                if layer_num == (start_num + 1):
                    current_layers_neurons = []
                    current_layers_neurons.append(neurons_num)
                elif layer_num == (end_num - 1):
                    current_layers_neurons.append(neurons_num)
                    current_results["results"].append(current_layers_neurons)
                else:
                    current_layers_neurons.append(neurons_num)
                
        results.append(current_results)
    
    
    output = []
    for config in results:
        avg = 0
        for result in config["results"]:
            avg += sum(result) / (num_neurons * (end_num - start_num - 1))
        output.append({
            "sparsity_name": config["sparsity_name"],
            "task_name": config["task_name"],
            "avg": avg / len(config["results"])
        })
    
    return output
    
    

def load_opt_param(name, param, start_num, end_num):
    num = 0
    if len(name.split('.'))>4:
        num = int(name.split('.')[3])                    
    if not (num>start_num and num<end_num and ('fc1' in name or 'fc2' in name)):
        param.data = param.data.to('cuda:0').half()

def load_llama_param(num, name, param, start_num, end_num):
    num = 0
    if len(name.split('.'))>3:
        num = int(name.split('.')[2])
    if not (num>start_num and num<end_num and ('gate_proj' in name or 'up_proj' in name or 'down_proj' in name)):
        param.data = param.data.to('cuda:0').half()

def load_model_memory_limit(checkpoint_path, start_num, end_num, model_name):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map='cpu')
    num_layers = model.config.num_hidden_layers
    for name, param in model.named_parameters():
        if ("opt" in model_name):
            load_opt_param(name, param, start_num, end_num)
        elif ("llama" in model_name):
            load_llama_param(name, param, start_num, end_num)
            
    return model, num_layers