import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_scheduler,
)
from tqdm import tqdm


class CategoryPredictor:
    """
    Fine-tunes BERT on Yahoo Answers Topics and provides predict(text) -> (label, confidence).


    Fixes:
    - Renames target column to 'labels' and REMOVES original dataset columns to avoid passing
    unexpected kwargs (e.g., 'topic') to the model's forward().
    - Builds robust input dict so only valid keys are passed to the model.
    """


    def __init__(self, model_name='bert-base-uncased', save_dir='outputs/checkpoints/bert_category_predictor', num_labels=None):
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)


        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # We'll set num_labels dynamically from the dataset if not provided
        self.num_labels = num_labels
        self.model = None


    def load_only(self, prefer_mps=True):
        """Load pre-trained model without training. Raises exception if not found."""
        ckpt_path = os.path.join(self.save_dir, 'model.pt')
        label_map_path = os.path.join(self.save_dir, 'label_map.json')

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"❌ Model checkpoint not found at {ckpt_path}\n"
                f"   Please train the model first using the notebook or call load_or_train()"
            )
        
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"❌ Label map not found at {label_map_path}\n"
                f"   Please train the model first using the notebook"
            )

        id2label = {int(k): v for k, v in json.load(open(label_map_path)).items()}
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
        # device select
        self.device = 'mps' if (prefer_mps and torch.backends.mps.is_available()) else 'cpu'
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        print(f"✅ Loaded Category Predictor from {ckpt_path} (device: {self.device})")

    def load_or_train(self, prefer_mps=True):
        ckpt_path = os.path.join(self.save_dir, 'model.pt')
        label_map_path = os.path.join(self.save_dir, 'label_map.json')

        if os.path.exists(ckpt_path) and os.path.exists(label_map_path):
            try:
                id2label = {int(k): v for k, v in json.load(open(label_map_path)).items()}
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(id2label),
                    id2label=id2label,
                    label2id={v: k for k, v in id2label.items()},
                )
                # device select
                self.device = 'mps' if (prefer_mps and torch.backends.mps.is_available()) else 'cpu'
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                self.model.to(self.device)
                print(f"✅ Loaded Category Predictor from {ckpt_path} (device: {self.device})")
                return
            except Exception as e:
                print(f"⚠️ Checkpoint load failed ({e}); retraining from scratch...")
                try:
                    os.remove(ckpt_path)
                except Exception:
                    pass

        # Train fresh
        self._train_yahoo(prefer_mps=prefer_mps)

    
    def _train_yahoo(self, epochs=3, lr=2e-5, batch_size=32, train_subset=50000, prefer_mps=True):
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        from transformers import get_scheduler, AdamW

        # Choose device safely
        use_mps = prefer_mps and torch.backends.mps.is_available()
        self.device = 'mps' if use_mps else 'cpu'

        ds = load_dataset('yahoo_answers_topics')
        train_ds = ds['train'].shuffle(seed=42)
        if train_subset is not None:
            train_ds = train_ds.select(range(min(train_subset, len(train_ds))))

        topic_feature = ds['train'].features['topic']
        topic_names = list(topic_feature.names)
        id2label = {i: name for i, name in enumerate(topic_names)}
        label2id = {name: i for i, name in id2label.items()}
        os.makedirs(self.save_dir, exist_ok=True)
        json.dump(id2label, open(os.path.join(self.save_dir, 'label_map.json'), 'w'))

        # (Re)build model with label maps
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(topic_names),
            id2label=id2label,
            label2id=label2id,
        ).to(self.device)

        def tokenize(batch):
            enc = self.tokenizer(
                batch['question_title'],
                truncation=True,
                padding='max_length',
                max_length=128,
            )
            enc['labels'] = batch['topic']  # ints 0..9 already
            return enc

        tokenized = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
        # Keep only what HF expects
        wanted_cols = ['input_ids', 'attention_mask', 'labels']
        for col in list(tokenized.features.keys()):
            if col not in wanted_cols:
                tokenized = tokenized.remove_columns(col)
        tokenized.set_format(type='torch')

        # pin_memory=False for MPS; persistent_workers=False avoids macOS worker issues
        loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True,
                            pin_memory=False, num_workers=0, persistent_workers=False)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(loader)
        scheduler = get_scheduler('linear', optimizer=optimizer,
                                num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            total_loss = 0.0
            for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
                # Move inputs explicitly and force correct dtype
                input_ids = batch['input_ids'].to(self.device, dtype=torch.long, non_blocking=False)
                attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long, non_blocking=False)
                labels = batch['labels'].to(self.device, dtype=torch.long, non_blocking=False)

                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()


        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pt'))


    def predict(self, text: str):
        if self.model is None:
            self.load_or_train()
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
            conf = float(probs[pred_id].item())


        # map id back to label
        id2label_path = os.path.join(self.save_dir, 'label_map.json')
        if os.path.exists(id2label_path):
            id2label = {int(k): v for k, v in json.load(open(id2label_path)).items()}
            return id2label[pred_id], conf
        return str(pred_id), conf
