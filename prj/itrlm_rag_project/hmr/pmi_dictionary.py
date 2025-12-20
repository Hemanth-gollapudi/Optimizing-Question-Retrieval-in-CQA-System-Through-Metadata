import math
import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm


class PMIDictionary:
    def __init__(self, cfg_model, save_dir='outputs/pmi_dicts'):
        self.alpha = cfg_model.get('alpha', 0.3)
        self.top_n = cfg_model.get('top_n_translations', 100)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)


    def _calc_mutual_information(self, cooccur, q_count, a_count, total_pairs):
        mi = {}
        for (wq, wa), c in tqdm(cooccur.items(), desc='PMI calculation'):
            p_wq_wa = c / total_pairs
            p_wq = q_count[wq] / total_pairs
            p_wa = a_count[wa] / total_pairs
            val = math.log((p_wq_wa / (p_wq * p_wa)) + 1e-12)
            mi[(wq, wa)] = val
        return mi


    def _normalize(self, mi):
        norm_dict = defaultdict(list)
        for (wq, wa), score in mi.items():
            norm_dict[wq].append((wa, score))


        for wq in norm_dict:
            top_pairs = sorted(norm_dict[wq], key=lambda x: -x[1])[:self.top_n]
            total = sum(max(s, 0.0) for _, s in top_pairs) + 1e-12
            norm_dict[wq] = [(wa, max(s, 0.0) / total) for wa, s in top_pairs]
        return norm_dict


    def build(self, qa_pairs, output_file='pmi_general.json'):
        print(f"[PMI] Building general dictionary from {len(qa_pairs)} Q-A pairs...")


        q_count = Counter()
        a_count = Counter()
        cooccur = Counter()


        for q, a in tqdm(qa_pairs, desc='Counting co-occurrences'):
            q_words = q.split()
            a_words = a.split()
            for wq in set(q_words):
                q_count[wq] += 1
                for wa in set(a_words):
                    a_count[wa] += 1
                    cooccur[(wq, wa)] += 1


        total_pairs = sum(cooccur.values())
        mi = self._calc_mutual_information(cooccur, q_count, a_count, total_pairs)
        norm = self._normalize(mi)

        path = os.path.join(self.save_dir, output_file)
        json.dump(norm, open(path, 'w'))
        print(f"✅ PMI dictionary saved to {path}")


    def build_by_category(self, qa_pairs_by_cat: dict):
        for cat, pairs in qa_pairs_by_cat.items():
            file = f"pmi_{cat.lower().replace(' ', '_')}.json"
            self.build(pairs, output_file=file)


    def load(self, path):
        with open(path, 'r') as f:
            return json.load(f)