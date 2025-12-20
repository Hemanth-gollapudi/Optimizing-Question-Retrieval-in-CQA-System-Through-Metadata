import os
import torch
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
)


class RAGAnswerGen:
    def __init__(self, cfg_rag, index_path='outputs/faiss_index.index'):
        self.embed_model = SentenceTransformer(cfg_rag['embed_model'])
        self.gen_model_name = cfg_rag['gen_model']
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.top_k_ctx = cfg_rag.get('top_k_ctx', 5)
        self.max_tokens = cfg_rag.get('max_answer_tokens', 50)

        print(f"[RAG] Using device: {self.device}")


        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
        if "t5" in self.gen_model_name or "flan" in self.gen_model_name:
            print(f"[RAG] Using Seq2SeqLM model: {self.gen_model_name}")
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name).to(self.device)
        else:
            print(f"[RAG] Using CausalLM model: {self.gen_model_name}")
            self.gen_model = AutoModelForCausalLM.from_pretrained(self.gen_model_name).to(self.device)
        # self.gen_model = AutoModelForCausalLM.from_pretrained(self.gen_model_name).to(self.device)


        self.index_path = index_path
        self.index = None
        self.ctx_texts = []


    def build_faiss(self, answers):
        print(f"[RAG] Building FAISS index for {len(answers)} candidate answers...")
        embeddings = self.embed_model.encode(answers, convert_to_numpy=True, show_progress_bar=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        self.ctx_texts = answers
        
        # Save context texts to a JSON file
        ctx_texts_path = self.index_path.replace('.index', '_texts.json')
        with open(ctx_texts_path, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        
        print(f"✅ FAISS index built and saved at {self.index_path}")
        print(f"✅ Context texts saved at {ctx_texts_path}")


    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✅ Loaded FAISS index from {self.index_path}")
            
            # Load context texts from JSON file
            ctx_texts_path = self.index_path.replace('.index', '_texts.json')
            if os.path.exists(ctx_texts_path):
                with open(ctx_texts_path, 'r', encoding='utf-8') as f:
                    self.ctx_texts = json.load(f)
                print(f"✅ Loaded {len(self.ctx_texts)} context texts from {ctx_texts_path}")
            else:
                print(f"⚠️ Warning: Context texts file not found at {ctx_texts_path}")
                print(f"⚠️ The FAISS index will not work without context texts.")
                raise FileNotFoundError(f"Context texts file not found: {ctx_texts_path}")
        else:
            raise FileNotFoundError("Index not found; please build it first.")


    def retrieve_context(self, question, k=None):
        if self.index is None:
            self.load_index()
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k or self.top_k_ctx)
        return [self.ctx_texts[i] for i in I[0]]


    def generate(self, question):
        context = self.retrieve_context(question)
        prompt = f"Context: {' '.join(context)}\nQuestion: {question}\nGenerate a concise relevant answer (<50 tokens):"

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        output = self.gen_model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=True, temperature=0.7)
        gen_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return gen_text.split('Question:')[-1].strip()