from random import choices
#from mistralai import Mistral
import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re 
from collections import Counter
#from sentence_transformers import SentenceTransformer
#import faiss
import numpy as np
import json
import os
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy import sparse


class FatwaRetriever:
    def __init__(self, folder_path, embedding_model="intfloat/multilingual-e5-large", cache_dir="cache", method='tfidf'):
        self.folder_path = folder_path
        self.cache_dir = cache_dir
        self.method = method  # "transformer" or "tfidf"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.device = self._select_device()
        print(f"✅ Using device: {self.device}")

        #self.model = SentenceTransformer(embedding_model, device=str(self.device))  # <-- Moved here

        self.fatwas = self._load_fatwas_from_folder(folder_path)
        self.questions = [f["Question"] for f in self.fatwas]
        self.answers = [f["Answer"] for f in self.fatwas]
        self.fatwas_hash = self._compute_data_hash(self.questions)

        if self.method == "transformer":
            if self._is_cache_valid():
                print("Loading embeddings and index from cache...")
                self.embeddings = np.load(os.path.join(self.cache_dir, "embeddings.npy"))
                with open(os.path.join(self.cache_dir, "questions.json"), "r", encoding="utf-8") as f:
                    self.questions = json.load(f)
                self.index = faiss.read_index(os.path.join(self.cache_dir, "faiss.index"))
            else:
                print("Generating transformer embeddings and index...")
                self.embeddings = self.model.encode(self.questions, show_progress_bar=True)
                self.index = self._build_faiss_index(self.embeddings)
                self._save_cache()
        elif self.method == "tfidf":
            if self._is_cache_valid(method="tfidf"):
                print("Loading TF-IDF matrix from cache...")
                with open(os.path.join(self.cache_dir, "tfidf_vectorizer.pkl"), "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.tfidf_matrix = np.load(os.path.join(self.cache_dir, "tfidf_matrix.npy"), allow_pickle=True)
                with open(os.path.join(self.cache_dir, "questions.json"), "r", encoding="utf-8") as f:
                    self.questions = json.load(f)
            else:
                print("Building TF-IDF matrix...")
                self.vectorizer = TfidfVectorizer()
                self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
                self._save_cache(method="tfidf")
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")


    def _select_device(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    torch.ones(1).to(f"cuda:{i}")
                    return torch.device(f"cuda:{i}")
                except RuntimeError:
                    continue
        return torch.device("cpu")

    def _load_fatwas_from_folder(self, folder_path):
        all_fatwas = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                full_path = os.path.join(folder_path, filename)
                with open(full_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_fatwas.extend(data)
                        elif isinstance(data, dict):
                            all_fatwas.append(data)
                    except json.JSONDecodeError:
                        print(f"Skipped invalid JSON file: {filename}")
        return all_fatwas

    def _compute_data_hash(self, texts):
        m = hashlib.md5()
        for text in texts:
            m.update(text.encode("utf-8"))
        return m.hexdigest()

    def _is_cache_valid(self, method="transformer"):
        hash_file = os.path.join(self.cache_dir, f"hash_{method}.txt")
        if not os.path.exists(hash_file):
            return False
        with open(hash_file, "r") as f:
            return f.read().strip() == self.fatwas_hash
    
    def _save_cache(self, method="transformer"):
        if method == "transformer":
            np.save(os.path.join(self.cache_dir, "embeddings.npy"), self.embeddings)
            with open(os.path.join(self.cache_dir, "questions.json"), "w", encoding="utf-8") as f:
                json.dump(self.questions, f, ensure_ascii=False, indent=2)
            faiss.write_index(self.index, os.path.join(self.cache_dir, "faiss.index"))
            with open(os.path.join(self.cache_dir, "hash_transformer.txt"), "w") as f:
                f.write(self.fatwas_hash)
                print(f"Cache saved with hash: {self.fatwas_hash}")
                print(f"Cache directory: {self.cache_dir}")
        elif method == "tfidf":
            with open(os.path.join(self.cache_dir, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.vectorizer, f)
            np.save(os.path.join(self.cache_dir, "tfidf_matrix.npy"), self.tfidf_matrix)
            with open(os.path.join(self.cache_dir, "questions.json"), "w", encoding="utf-8") as f:
                json.dump(self.questions, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.cache_dir, "hash_tfidf.txt"), "w") as f:
                f.write(self.fatwas_hash)
            print(f"Cache saved for {method} with hash: {self.fatwas_hash}")

    
    def _build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        return index
    
    
    def retrieve(self, query, top_k=1):
        if self.method == "transformer":
            query_embedding = self.model.encode([query])
            D, I = self.index.search(np.array(query_embedding), top_k)
        elif self.method == "tfidf":
            query_vec = self.vectorizer.transform([query])
            similarities = self.tfidf_matrix.dot(query_vec.T).toarray().squeeze()
            I = np.argsort(similarities)[::-1][:top_k]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.format_retrieved_documents(
    [(self.questions[i], self.answers[i]) for i in I.flatten()]
)

    
    def format_retrieved_documents(self, retrieved_docs):
        """
        Format retrieved fatwas in a numbered list.
        """
        return [f"\nFatwa {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)]

 

class CollaborativeLLms:
    
    def __init__(self, models):
        """
        models: list of LLM objects, each must have a .query(text) method
        """
        # Pick the first free GPU (or fall back to CPU)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    # Quick dummy allocation to check if the GPU is really free
                    torch.ones(1).to(f"cuda:{i}")
                    self.device = torch.device(f"cuda:{i}")
                    break
                except RuntimeError:  # GPU is busy or OOM
                    continue
            else:  # no GPU was free
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}") 
        self.models = models

    def query_all(self, prompt):
        """
        Query all models with optional fatwa retrieval context.
        """
        return [model["model"].query(prompt) for model in self.models]
    
    def batch_query(self, csv_path, output_csv="predictions.csv", nbr_rows=None, retriever=None):
        """
        Iterates over a CSV file and queries all models for each row's text.
        Saves a DataFrame with columns: id_question, response (for each model) incrementally.
        """
        if nbr_rows is not None:
            df = pd.read_csv(csv_path, nrows=nbr_rows)
        else:
            df = pd.read_csv(csv_path)[500:]
        model_names = [m["model_name"] for m in self.models]

        results = []
        for idx, row in df.iterrows():
            context_text = ""
            if retriever:
                context_chunks = retriever.retrieve(row['question'])
                context_text = "\n".join(context_chunks)
            #print(f"Context text: {context_text}")
            prompt = self.generate_mcq_prompt(row, context_text=context_text)
            #print(f'Prompt: {prompt}')
            responses = self.query_all(prompt)
            result = {
                "id_question": row.get('id_question', idx),  # fallback to idx if no question_id
            }
            for name, response in zip(model_names, responses):
                result[name] = response
            results.append(result)

            # Export progress incrementally after each row:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False)
            print(f"Processed row {idx}: Responses: {responses}")
            print(f'Saved intermediate results to {output_csv}')


        print(f"Predictions saved to {output_csv}")
        return results_df

    def clean_responses(self, csv_path, output_csv=None):
        df = pd.read_csv(csv_path)
        def extract_letter(resp):
            # Try to find a single uppercase letter A-F, possibly after "Answer:" or at the start
            match = re.search(r"\b([A-F])\b", str(resp))
            if match:
                return match.group(1)
            # Try to find at the very start (e.g., "A) ...")
            match = re.match(r"([A-F])\)", str(resp).strip())
            if match:
                return match.group(1)
            return ""  # or np.nan if you prefer

        for col in df.columns:
            if col != 'id_question':
                df[col] = df[col].apply(extract_letter)
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df
    
    def generate_mcq_prompt(self, csv_row, context_text=""):
        """
        Generate MCQ prompt with few-shot Islamic examples.
        """
        question = csv_row['question']
        options = []
        for idx, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F'], start=1):
            option_col = f'option{idx}'
            if option_col in csv_row and pd.notna(csv_row[option_col]):
                value = str(csv_row[option_col]).strip()
                # Only add letter if not already present
                if not value.startswith(f"{letter})"):
                    options.append(f"{letter}) {value}")
                else:
                    options.append(value)
        options_text = "\n".join(options)
        valid_letters = "/".join([letter for letter in ['A', 'B', 'C', 'D', 'E', 'F'] if f'option{ord(letter)-64}' in csv_row])

        few_shot_examples = """
You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option.
        """
        
        if context_text != "":
            context_text = "\nThe following fatwas may assist you:\n" + context_text
       
        prompt = f"""{few_shot_examples}
{context_text}\n
Now answer the following question:\n
Question:\n {question} \n
{options_text} \n
Please respond using **only one English letter** from the following: {valid_letters}
Do not write any explanation or additional text.
                """

        #print(f"Generated prompt: {prompt}")  # Debugging line to check the generated prompt
        
        return prompt
    
    def generate_mcq_prompt_v2(self, csv_row):
        """
        Generate MCQ prompt with few-shot Islamic examples.
        """
        question = csv_row['question']
        options = []
        for idx, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F'], start=1):
            option_col = f'option{idx}'
            if option_col in csv_row and pd.notna(csv_row[option_col]):
                value = str(csv_row[option_col]).strip()
                # Only add letter if not already present
                if not value.startswith(f"{letter})"):
                    options.append(f"{letter}) {value}")
                else:
                    options.append(value)
        options_text = "\n".join(options)
        valid_letters = "/".join([letter for letter in ['A', 'B', 'C', 'D', 'E', 'F'] if f'option{ord(letter)-64}' in csv_row])

        few_shot_examples = """
    Example 1:
    Question: ما مدة المسح على الخفين للمقيم؟
    A) يوم وليلة
    B) ثلاثة أيام بلياليهن
    C) يومان وليلتان
    D) أسبوع كامل
    Answer: A

    Example 2:
    Question: توفي عن أب، وأخوين شقيقين، وابن أخ شقيق، وعمين شقيقين، وأم، وبنتين، و زوجة، فما نصيب الأم؟
    A) الثلث
    B) الربع
    C) السدس
    D) الثمن
    E) النصف
    F) لا شيء
    Answer: C

    Now answer the following question:
    """

        prompt = f"""{few_shot_examples}

    You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option.

    Question: {question}

    {options_text}

    Please respond using **only one English letter** from the following: {valid_letters}
    Do not write any explanation or additional text.
    """
        print(f"Generated prompt: {prompt}")  # Debugging line to check the generated prompt
        return prompt
    
    def evaluate_models(self, prediction_csv, reference_csv, output_csv=None):
        """
        Evaluate accuracy for each model column in predictions.csv.
        Keeps all columns from predictions.csv and adds 'label' from reference.csv.
        Optionally writes merged data to output_csv.
        """
        # Load data
        preds_df = pd.read_csv(prediction_csv)
        ref_df = pd.read_csv(reference_csv)

        # Validate structure
        assert 'id_question' in preds_df.columns, "'id_question' missing in predictions"
        assert 'id_question' in ref_df.columns, "'id_question' missing in reference"
        assert 'label' in ref_df.columns, "'label' column missing in reference"

        # Perform inner join, keep all columns from predictions.csv + 'label'
        merged_df = preds_df.merge(ref_df[['id_question', 'label']], on='id_question', how='left')

        print(merged_df)

        # Optional: export merged file
        if output_csv:
            merged_df.to_csv(output_csv, index=False)
            print(f"Merged file saved to {output_csv}")

        # Evaluate accuracy for all model columns except 'id_question'
        model_columns = [col for col in preds_df.columns if col != 'id_question']
        accuracies = {}
        
        print(model_columns)

        for model in model_columns:
            preds = merged_df[model].astype(str).str.strip().str.upper()
            truth = merged_df['label'].astype(str).str.strip().str.upper()
            total = len(truth)
            correct = (preds == truth).sum()
            acc = correct / total if total > 0 else 0
            accuracies[model] = acc
            print(f"Accuracy for {model}: {acc:.2%} ({correct}/{total})")

        return accuracies
    

    def vote_majority(self, input_csv="predictions.csv", output_csv="voted_predictions.csv"):
        """
        Reads predictions from multiple models and outputs a file with majority vote response.
        If each model gives a different answer, uses the last model's answer.
        """
        df = pd.read_csv(input_csv)
        model_columns = [col for col in df.columns if col != 'id_question']

        voted_rows = []
        for idx, row in df.iterrows():
            id_question = row['id_question']
            responses = [str(row[col]).strip() for col in model_columns if pd.notna(row[col])]
            valid_responses = [r for r in responses if r in ['A', 'B', 'C', 'D', 'E', 'F']]

            if valid_responses:
                counts = Counter(valid_responses)
                most_common = counts.most_common()
                if len(most_common) == 1:
                    majority_vote = most_common[0][0]
                elif most_common[0][1] > most_common[1][1]:
                    majority_vote = most_common[0][0]
                else:
                    # Tie or all different: take last model’s response if valid
                    last_response = str(row[model_columns[-1]]).strip()
                    majority_vote = last_response if last_response in ['A', 'B', 'C', 'D', 'E', 'F'] else ""
            else:
                majority_vote = ""

            voted_rows.append({"id_question": id_question, "response": majority_vote})

        voted_df = pd.DataFrame(voted_rows)
        voted_df.to_csv(output_csv, index=False)
        print(f"Voted predictions saved to {output_csv}")
        return voted_df

    
    
    def add_model_from_hf(self, model_name, use_api=False):
        """
        Downloads a model and tokenizer from Hugging Face and adds it to the models list.
        The model object must have a .query(text) method.
        """
        if model_name == "Qwen/Qwen3-1.7B":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Qwen3 model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )

            class QwenModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    
                def qwen_clean_response(self, response):
                    """
                    Extracts the first standalone uppercase letter A-F from a text response.
                    """
                    # Try: explicit **Answer:** line
                    match = re.search(r"\*\*Answer:\*\*\s*([A-F])", response)
                    if match:
                        return match.group(1)

                    # Fallback: any standalone uppercase letter A-F
                    match = re.search(r"\b([A-F])\b", response)
                    if match:
                        return match.group(1)

                    return ""  # or return None if you prefer

                def query(self, prompt):
                    messages = [{"role": "user", "content": prompt}]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=10000,  # or a sensible number
                    )

                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                    try:
                        # Find the position of 151668 (</think> token ID)
                        think_index = output_ids.index(151668) + 1  # +1 to get tokens *after* </think>
                        relevant_output_ids = output_ids[think_index:]
                        #print(f"Found </think> at index {think_index}, extracting relevant output.")
                    except ValueError:
                        # </think> not found: fallback to entire output
                        relevant_output_ids = output_ids

                    # Decode only the relevant part
                    decoded_text = self.tokenizer.decode(relevant_output_ids, skip_special_tokens=True).strip()

                    #print(f"Decoded text after </think>: {decoded_text}")

                    # Extract actual A-F letter
                    cleaned_response = self.qwen_clean_response(decoded_text)
                    return cleaned_response
                
                def query_old(self, prompt):
                    
                    print(f"Querying model")
                    print(self.tokenizer.decode([151668]))
                    
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=200
                    )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                    try:
                        think_index = output_ids.index(151668) + 1  
                        relevant_output_ids = output_ids[think_index:]
                    except ValueError:
                        relevant_output_ids = output_ids

                    decoded_text = self.tokenizer.decode(relevant_output_ids, skip_special_tokens=True).strip()

                    print(f"Decoded text after </think>: {decoded_text}")

                    # Extract actual A-F letter
                    cleaned_response = self.qwen_clean_response(decoded_text)
                    return cleaned_response

            model_detail = {
                "model_name": model_name,
                "model": QwenModelWrapper(model, tokenizer),
            }
            
            self.models.append(model_detail)
            
        elif model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # New DeepSeek branch
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype='auto',
                device_map=self.device,
            )

            class DeepSeekModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    #self.tokenizer.pad_token = self.tokenizer.eos_token  # Explicit fix

                def query(self, prompt):
                    messages = [
                        {"role": "user", "content": prompt},
                    ]
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(self.model.device)

                    outputs = self.model.generate(**inputs, max_new_tokens=100000)
                    response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
                    return response

            model_detail = {
                "model_name": model_name,
                "model": DeepSeekModelWrapper(model, tokenizer),
            }
            self.models.append(model_detail)
        
        elif model_name.startswith("meta-llama/"):
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=""
            )

            class LlamaModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                def query(self, text):
                    # We mimic your exact standalone code:
                    messages = [
                        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                        {"role": "user", "content": text},
                    ]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(self.model.device)

                    terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            max_new_tokens=256,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
                        )
                    response = outputs[0][input_ids.shape[-1]:]
                    return self.tokenizer.decode(response, skip_special_tokens=True)

            model_detail = {
                "model_name": model_name,
                "model": LlamaModelWrapper(model, tokenizer),
            }
            self.models.append(model_detail)
        
        elif model_name == "NousResearch/DeepHermes-3-Llama-3-8B-Preview":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )

            class DeepHermesModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                def query(self, prompt):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=32768
                    )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    try:
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0
                    #thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    return content

            model_detail = {
                "model_name": model_name,
                "model": DeepHermesModelWrapper(model, tokenizer),
            }
            self.models.append(model_detail)    
        
        elif model_name == "moonshotai/Kimi-K2-Base":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                
            )
            class KimiK2ModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                def query(self, prompt):
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors='pt'
                    ).to(self.model.device)

                    generated_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=2500,
                        temperature=0.8,
                        repetition_penalty=1.1,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
                    return response

            model_detail = {
                "model_name": model_name,
                "model": KimiK2ModelWrapper(model, tokenizer),
            }
            self.models.append(model_detail)
        
        
        elif model_name == "openai/gpt-oss-20b":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )

            class GPTOSS20BModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                def query(self, prompt):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    try:
                        # Apply chat template and move to device
                        inputs = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(self.model.device)

                        # Generate response
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                        # Decode only the new tokens (assistant's reply)
                        input_len = inputs["input_ids"].shape[-1]
                        full_output = outputs[0]
                        response = self.tokenizer.decode(full_output[input_len:], skip_special_tokens=True)
                        return response.strip()

                    except Exception as e:
                        print(f"Error generating response from {model_name}: {e}")
                        return None

            model_detail = {
                "model_name": model_name,
                "model": GPTOSS20BModelWrapper(model, tokenizer),
            }
            self.models.append(model_detail)
        
        
        elif model_name == "ALLaM-AI/ALLaM-7B-Instruct-preview":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            class ALLaM7BInstructPreviewModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                def query(self, prompt):
                    import re
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]

                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=100000, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
                    )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    prediction = response.split(prompt)[-1].strip()
                    match = re.search(r"\b([A-F])\b", prediction.upper())
                    if match:
                        return match.group(1)

                    return None

            model_detail = {
                "model_name": model_name,
                "model": ALLaM7BInstructPreviewModelWrapper(model, tokenizer),
            }
            self.models.append(model_detail)
        
        elif model_name == "QCRI/Fanar-1-9B":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import re
            import requests
            
            
            if not use_api:
                # Load tokenizer and model (local version)
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                )
            else:
                # use api endpoint
                model = None
                tokenizer = None
            
            def pack_choices(choice1, choice2, choice3, choice4, choice5=None, choice6=None):
                """Pack MCQ choices as list of (letter, text)."""
                choices = [("A", choice1), ("B", choice2), ("C", choice3), ("D", choice4)]
                if choice5: choices.append(("E", choice5))
                if choice6: choices.append(("F", choice6))
                return choices
            
            
            def get_valid_responses(choice5, choice6):
                """Generate valid response set."""
                valid = {"A", "B", "C", "D"}
                if choice5: valid.add("E")
                if choice6: valid.add("F")
                return valid
            
            
            def clean_and_validate_response(raw_response, valid_responses):
                """Clean and extract valid answer letter."""
                if not raw_response:
                    return None

                raw_response = raw_response.strip().upper()

                # More robust regex extraction
                match = re.search(r"(?:answer\s*(?:is)?\s*[:\-]?\s*)([A-F])", raw_response, re.IGNORECASE)
                if match:
                    candidate = match.group(1).upper()
                    if candidate in valid_responses:
                        return candidate

                # Fallback simple extraction
                match = re.search(r"\b([A-F])\b", raw_response)
                if match and match.group(1) in valid_responses:
                    return match.group(1)

                return None
            
            
            def generate_mcq_prompt(question, choices):
                """
                Generate MCQ prompt with few-shot Islamic examples.
                """
                options_text = "\n".join([f"{letter}) {text}" for letter, text in choices])
                valid_letters = "/".join([letter for letter, _ in choices])

                few_shot_examples = """
                Example 1:
                Question: ما مدة المسح على الخفين للمقيم؟
                A) يوم وليلة
                B) ثلاثة أيام بلياليهن
                C) يومان وليلتان
                D) أسبوع كامل
                Answer: A
                
                Example 2:
                Question: توفي عن أب، وأخوين شقيقين، وابن أخ شقيق، وعمين شقيقين، وأم، وبنتين، و زوجة، فما نصيب الأم؟
                A) الثلث
                B) الربع
                C) السدس
                D) الثمن
                E) النصف
                F) لا شيء
                Answer: C
                
                Now answer the following question:
            """

                prompt = f"""{few_shot_examples}

            You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option.

            Question: {question}

            {options_text}

            Please respond using **only one English letter** from the following: {valid_letters}
            Do not write any explanation or additional text.
            """
                return prompt
            
            # Define the external Fanar API prediction function.
            # (Make sure pack_choices, get_valid_responses, generate_mcq_prompt and clean_and_validate_response exist.)
            def get_prediction_fanar(question, choice1, choice2, choice3, choice4, choice5=None, choice6=None,
                                    model_version="Islamic-RAG", max_retries=3):
                """Inference using Fanar API."""
                
                fanar_api_key = ""
                if not fanar_api_key:
                    print("Fanar API key missing.")
                    return None

                fanar_url = "https://api.fanar.qa/v1/chat/completions"
                choices = pack_choices(choice1, choice2, choice3, choice4, choice5, choice6)
                valid_responses = get_valid_responses(choice5, choice6)
                prompt_api = generate_mcq_prompt(question, choices)  # create the API prompt format

                headers = {"Authorization": f"Bearer {fanar_api_key}", "Content-Type": "application/json"}
                data = {"model": model_version, "messages": [{"role": "user", "content": prompt_api}]}

                for attempt in range(1, max_retries + 1):
                    try:
                        response = requests.post(fanar_url, json=data, headers=headers)
                        response_json = response.json()

                        if response.status_code == 200:
                            raw_result = response_json["choices"][0]["message"]["content"].strip().upper()
                            cleaned_result = clean_and_validate_response(raw_result, valid_responses)
                            if cleaned_result:
                                print(f"✅ {question} | Fanar  | Prediction: {cleaned_result}")
                                return cleaned_result
                        else:
                            print(f"❌ Fanar API Error: {response.text}")
                            return None
                    except Exception as e:
                        print(f"❌ Fanar Error: {e}")
                        return None

                print("❌ Failed after retries.")
                return None

            class Fanar1_9B_ModelWrapper:
                def __init__(self, model, tokenizer, use_api=use_api):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.use_api = use_api  # if True, use the external API for inference

                def query(self, prompt):
                    if self.use_api:
                        # Extract question and options from the prompt.
                        # Here we assume the prompt is built via your generate_mcq_prompt (or v2) method.
                        # For example, if the prompt looks like:
                        #
                        #    "Question: <question text>\nA) ...\nB) ...\nC) ...\nD) ...\n..."
                        #
                        # then we can split by lines.
                        lines = prompt.splitlines()
                        question = ""
                        options = []
                        for line in lines:
                            if line.startswith("Question:"):
                                question = line.replace("Question:", "").strip()
                            elif re.match(r"^[A-F]\)", line):
                                # e.g., "A) option text"
                                parts = line.split(")", 1)
                                if len(parts) == 2:
                                    options.append(parts[1].strip())
                        # Ensure there are at least 4 options
                        if len(options) < 4:
                            print("Not enough options extracted for Fanar API.")
                            return None
                        # Call the external Fanar API function. (Pass first 4–6 options as needed)
                        return get_prediction_fanar(question, options[0], options[1], options[2], options[3],
                                                    options[4] if len(options) > 4 else None,
                                                    options[5] if len(options) > 5 else None)
                    else:
                        # Local inference using the model’s chat template.
                        messages = [{"role": "user", "content": prompt}]
                        try:
                            inputs = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_tensors="pt",
                                return_dict=True
                            ).to(self.model.device)
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=512,
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                            response_parts = full_response.split("<start_of_turn>model")
                            if len(response_parts) < 2:
                                return None
                            assistant_response = response_parts[1].strip().split("<end_of_turn>")[0].strip()
                            match = re.search(r"\b([A-F])\b", assistant_response.upper())
                            if match:
                                return match.group(1)
                            return assistant_response
                        except Exception as e:
                            print(f"Error during Fanar local query: {e}")
                            return None

            model_detail = {
                "model_name": model_name,
                "model": Fanar1_9B_ModelWrapper(model, tokenizer, use_api=True),  # set use_api=True to use the API method
            }
            self.models.append(model_detail)
            
        elif model_name.startswith("gemini-2.5-flash"):
            from google import genai

            # Initialize Gemini API client
            client = genai.Client(
                #api_key=""
                api_key=""
            )

            class GeminiModelWrapper:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name

                def query(self, prompt):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt
                        )
                        text = response.text.strip()
                        #print(f"Raw Gemini response: {text}")

                        # Optional: clean to extract single letter A-F
                        match = re.search(r"\b([A-F])\b", text)
                        if match:
                            return match.group(1)
                        return text  # fallback to raw if no letter found

                    except Exception as e:
                        print(f"❌ Gemini API error: {e}")
                        return ""

            model_detail = {
                "model_name": model_name,
                "model": GeminiModelWrapper(client, model_name),
            }
            self.models.append(model_detail)
        
        elif model_name.startswith("gpt-"):
            import openai

            openai.api_key = ""  # Replace with your actual key

            class OpenAIModelWrapper:
                def __init__(self, model_name):
                    self.model_name = model_name

                def query(self, prompt):
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3
                        )
                        text = response.choices[0].message["content"].strip()

                        # Optional: extract letter A-F
                        match = re.search(r"\b([A-F])\b", text)
                        if match:
                            return match.group(1)
                        return text
                    except Exception as e:
                        print(f"❌ OpenAI API error: {e}")
                        return ""

            model_detail = {
                "model_name": model_name,
                "model": OpenAIModelWrapper(model_name),
            }
            self.models.append(model_detail)
        
        elif model_name == "gemini-1.5-pro":
            import google.generativeai as genai

            genai.configure(api_key="")  # Replace with your actual key

            class GeminiModelWrapper:
                def __init__(self, model_name):
                    self.model = genai.GenerativeModel(model_name)

                def query(self, prompt):
                    try:
                        response = self.model.generate_content(prompt)
                        text = response.text.strip()

                        # Optional: extract letter A-F
                        match = re.search(r"\b([A-F])\b", text)
                        if match:
                            return match.group(1)
                        return text
                    except Exception as e:
                        print(f"❌ Gemini API error: {e}")
                        return ""

            model_detail = {
                "model_name": model_name,
                "model": GeminiModelWrapper(model_name),
            }
            self.models.append(model_detail)
        
        elif model_name == "mistral-large-latest":
            from mistralai import Mistral
            import os
            api_key = ""
            model = "mistral-large-latest"

            client = Mistral(api_key=api_key)

            class MistralModelWrapper:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name

                def query(self, prompt):
                    try:
                        chat_response = self.client.chat.complete(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        text = chat_response.choices[0].message.content.strip()
                        #print(f"Raw Mistral response: {text}")

                        # Optional: extract single letter A-F if present
                        match = re.search(r"\b([A-F])\b", text)
                        if match:
                            return match.group(1)
                        return text  # fallback to raw if no letter found

                    except Exception as e:
                        print(f"❌ Mistral API error: {e}")
                        return ""

            model_detail = {
                "model_name": model_name,
                "model": MistralModelWrapper(client, model_name),
            }
            self.models.append(model_detail)
        
        elif model_name.startswith("deepseek-"):
            from openai import OpenAI

            # Initialize DeepSeek API client
            client = OpenAI(
                api_key="",  
                base_url="https://api.deepseek.com"
            )

            class DeepSeekModelWrapper:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name

                def query(self, prompt):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option. Please respond using **only one English letter** from the following: A, B, C, D, E, or F. Do not write any explanation or thinking or additional text."},
                                {"role": "user", "content": prompt}
                            ],
                        )
                        print(f'Message: {response}')
                        text = response.choices[0].message.content.strip()

                        # Optional: clean to extract single letter A–F if needed
                        match = re.search(r"\b([A-F])\b", text)
                        if match:
                            return match.group(1)
                        return text

                    except Exception as e:
                        print(f"DeepSeek API error: {e}")
                        return ""

            model_detail = {
                "model_name": model_name,
                "model": DeepSeekModelWrapper(client, model_name),
            }
            self.models.append(model_detail)

        
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Qwen3 model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
                )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )

            class HFModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer

                def hf_clean_response(self, response):
                    """
                    Extracts the first standalone uppercase letter A-F from a text response.
                    """
                    # Try: explicit **Answer:** line
                    match = re.search(r"\*\*Answer:\*\*\s*([A-F])", response)
                    if match:
                        return match.group(1)

                    # Fallback: any standalone uppercase letter A-F
                    match = re.search(r"\b([A-F])\b", response)
                    if match:
                        return match.group(1)

                    return ""  # or return None if you prefer

                def query(self, prompt):
                    messages = [{"role": "user", "content": prompt}]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=32768,  # or a sensible number
                        temperature=0.6,
                        top_p=0.94,
                    )

                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    try:
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0
                    #thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    print(f"Content: {content}")

                    cleaned_response = self.hf_clean_response(content)
                    print(f"Cleaned response: {cleaned_response}")
                    return cleaned_response

            model_detail = {
                "model_name": model_name,
                "model": HFModelWrapper(model, tokenizer),
            }
            
            self.models.append(model_detail)



if __name__ == "__main__":
    # Example usage
    data_validation = "../data/data/MCQs/Task1_MCQs_Dev.csv"
    data_test = "../data/data/MCQs/Test/Task1_MCQ_Test.csv"
    
    
    
    evaluation_data = "./qwen.csv"
    evaluation_data_allam = "./allam_cleaned.csv"
    evaluation_data_fanar = "./fanar.csv"

    path_to_evaluation = r""
    
    

    # Example usage:
    collab_llms = CollaborativeLLms(models=[])
    
    collab_llms.evaluate_models(
        evaluation_data_fanar,
        './../data/comp/test_labled.csv',
        
    )
    #collab_llms.add_model_from_hf("ALLaM-AI/ALLaM-7B-Instruct-preview")
    
    
    #collab_llms.add_model_from_hf("openai/gpt-oss-20b")
    
    #collab_llms.add_model_from_hf("NousResearch/DeepHermes-3-Llama-3-8B-Preview")
    #collab_llms.add_model_from_hf("gemini-2.5-flash")
    #collab_llms.add_model_from_hf("gemini-2.5-flash")
    #collab_llms.add_model_from_hf("deepseek-reasoner")
    #collab_llms.add_model_from_hf("deepseek-chat")
    #collab_llms.add_model_from_hf("gpt-4o")
    #collab_llms.add_model_from_hf("mistral-large-latest")
    #collab_llms.add_model_from_hf("QCRI/Fanar-1-9B", use_api=True)
    
    
    
    # 
    #collab_llms.add_model_from_hf("Qwen/Qwen3-1.7B")
    
    #collab_llms.batch_query(data_test, output_csv="chatgpt.csv", retriever=retriever)
    #collab_llms.batch_query(data_test, output_csv="fanar2.csv", retriever=None)
    
    
    
    #collab_llms.batch_query(data_validation, output_csv="predictions_500_deepseek_no_res.csv", retriever=retriever)
    #collab_llms.clean_responses("predictions_500_deepseek_no_res.csv", "cleaned_predictions2.csv")
    #collab_llms.vote_majority("./merged_predictions.csv", "voted_predictions.csv")
    #collab_llms.evaluate_models("cleaned_predictions2.csv", "../data/data/MCQs/Task1_MCQs_Dev.csv")


    
    
    
    
    
    #collab_llms.add_model_from_hf("PhysicsWallahAI/Aryabhata-1.0")
    
    
    #collab_llms.add_model_from_hf("moonshotai/Kimi-VL-A3B-Thinking")
    #collab_llms.add_model_from_hf("moonshotai/Moonlight-16B-A3B-Instruct")
    
    
    #collab_llms.add_model_from_hf("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    
    
    #collab_llms.add_model_from_hf("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    #collab_llms.add_model_from_hf("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    #collab_llms.add_model_from_hf("HPLT/hplt2c_ara_checkpoints")
