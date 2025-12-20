import { motion } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { ArrowDown } from 'lucide-react'

export default function ITRLMFlow() {
  const modules = [
    {
      title: '1. Probabilistic Dictionary Module',
      content: [
        'General Dictionary: Built from all QA pairs using normalized mutual information',
        'Category-Specific Dictionaries: 5 domains — Travel, Family & Relationships, Society & Culture, Business & Finance, Sports',
        'Translation Probability: P_MI(w_A|w_Q) = I(w_Q; w_A) / Σ I(w_Q; w_i)',
        'Top-10 translation candidates per word with normalized probabilities'
      ],
      color: 'from-blue-100 to-blue-50'
    },
    {
      title: '2. Category Prediction Module (BERT-based)',
      content: [
        'Input: Question text + Subject',
        'Model: Fine-tuned BERT multiclass classifier',
        'Training: 4M Yahoo! Answers question-category pairs',
        'Output: Predicted category for new question',
        'Accuracy: 77.92%',
        'Purpose: Select appropriate category-specific dictionary'
      ],
      color: 'from-emerald-100 to-emerald-50'
    },
    {
      title: '3. Question Expansion Module (GPT-4 + RAG)',
      content: [
        'Answer Generation Models: GPT-2 (FT), GPT-3 (FT), GPT-4 (base), GPT-4 + RAG',
        'RAG Implementation: Retrieve top-ranked candidate questions as context',
        'Generate answer using retrieved context + new question',
        'Best performance achieved with GPT-4 + RAG',
        'Expansion Process:',
        'Q_exp-new = Q_new + Generated Answer',
        'Q_exp-can = Q_can + Existing Answer'
      ],
      color: 'from-purple-100 to-purple-50'
    },
    {
      title: '4. ITRLM Ranking Module',
      content: [
        'Similarity Calculation: P(Q_exp-new | Q_exp-can) = γ·P(Q_new | Q_exp-can) + (1-γ)·P(genA | Q_exp-can)',
        'Translation Model: P_mix(w | Q_exp-can) = (1-β)·P_ml(w | Q_exp-can) + β·Σ P_MI(w|t)·P_ml(t|Q_exp-can)',
        'Parameters: γ=0.8, β=0.8, λ=0.2, α=0.3 (optimized via grid search)',
        'Output: Ranked list of candidate questions by similarity score'
      ],
      color: 'from-orange-100 to-orange-50'
    }
  ]

  return (
    <div className="flex flex-col items-center space-y-8 py-10">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-3xl font-bold text-gray-900 mb-4"
      >
        ITRLM Framework Architecture Flow
      </motion.h1>
      <p className="text-gray-700 text-center max-w-3xl mb-8">
        End-to-end architecture integrating probabilistic translation, BERT-based category prediction, GPT-4 + RAG question expansion, and ITRLM ranking.
      </p>

      {modules.map((mod, i) => (
        <motion.div
          key={mod.title}
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: i * 0.2 }}
          viewport={{ once: true }}
          className="flex flex-col items-center w-full"
        >
          <Card className={`w-[80%] bg-gradient-to-b ${mod.color} shadow-md border border-gray-200`}>
            <CardContent className="p-6">
              <h2 className="text-2xl font-semibold mb-3 text-gray-800">{mod.title}</h2>
              <ul className="list-disc pl-6 space-y-1 text-gray-700">
                {mod.content.map((line, idx) => (
                  <li key={idx}>{line}</li>
                ))}
              </ul>
            </CardContent>
          </Card>

          {i < modules.length - 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 + i * 0.2 }}
              className="my-4"
            >
              <ArrowDown className="w-8 h-8 text-gray-500 animate-bounce" />
            </motion.div>
          )}
        </motion.div>
      ))}

      <Card className="w-[75%] bg-gradient-to-b from-gray-100 to-gray-50 border border-gray-200 shadow-sm mt-6">
        <CardContent className="p-5">
          <h2 className="text-xl font-semibold mb-2 text-gray-800">Architecture Advantages</h2>
          <ul className="list-disc pl-6 text-gray-700 space-y-1">
            <li>Combines probabilistic methods with deep learning</li>
            <li>Domain-specific translation knowledge via BERT category prediction</li>
            <li>Enhanced semantic matching through GPT-4 + RAG expansion</li>
            <li>Interpretable and computationally efficient</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
