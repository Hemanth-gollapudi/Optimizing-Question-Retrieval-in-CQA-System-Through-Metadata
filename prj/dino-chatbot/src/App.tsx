// src/App.tsx
import { useState } from "react";
import { ChatInput } from "./components/ChatInput";
import { useSpeech } from "./hooks/useSpeech";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

interface LanguageDetectionResponse {
  detected_language: string;
  text: string;
}

interface TranslationResponse {
  translated_text: string;
  source_lang: string;
  target_lang: string;
}

interface GenerateAnswerResponse {
  question: string;
  answer: string;
}

interface Message {
  sender: "user" | "dino";
  text: string;
  mode?: "speech" | "text";
}

type ScreenState = "intro" | "chat";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "dino",
      text: "Hello, I'm Pebble. I synthesise research across languages—how may I assist you today?",
      mode: "text",
    },
  ]);
  const [screen, setScreen] = useState<ScreenState>("intro");
  const [activeLanguage, setActiveLanguage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const { speakText } = useSpeech();

  const handleStartChat = () => {
    setScreen("chat");
  };

const handleSend = async (text: string, mode: "speech" | "text", providedLanguage?: string) => {
    const trimmed = text.trim();
  if (!trimmed || isProcessing) {
    return;
  }
    setIsProcessing(true);
    try {
      const languageCode = providedLanguage ?? (await detectLanguageFromAPI(trimmed));
      setActiveLanguage(languageCode);

      const displayText = await translateText(trimmed, languageCode, languageCode);

      setMessages((prev) => [
        ...prev,
        {
          sender: "user",
          text: displayText,
          mode,
        },
      ]);

      const englishQuestion = await translateText(trimmed, languageCode, "en");

      const answerRes = await fetch(`${API_BASE_URL}/generate-answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: englishQuestion }),
      });
      if (!answerRes.ok) {
        throw new Error("Failed to generate answer");
      }
      const answerData: GenerateAnswerResponse = await answerRes.json();
      const finalAnswer = await translateText(answerData.answer, "en", languageCode);
      setMessages((prev) => [...prev, { sender: "dino", text: finalAnswer, mode: "text" }]);
      speakText(finalAnswer);
    } catch (error) {
      console.error(error);
      const fallback = "I lost connection to the fossil archives. Could you try again?";
      setMessages((prev) => [...prev, { sender: "dino", text: fallback, mode: "text" }]);
    } finally {
      setActiveLanguage(null);
      setIsProcessing(false);
    }
  };

  if (screen === "intro") {
    return (
      <main className="solo-viewer">
        <div className="intro-card">
          <img src="/image.avif" alt="Pebble the baby dinosaur" loading="lazy" />
          <div>
            <h1>Pebble Intelligence</h1>
            <p>Multilingual research assistant for precise, policy-grade answers.</p>
          </div>
          <button type="button" className="btn btn--primary" onClick={handleStartChat}>
            Launch workspace
          </button>
        </div>
      </main>
    );
  }

  return (
    <div className="app-background">
      <div className="app-shell">
        <section className="dino-panel">
          <div className="dino-panel__copy">
            <p className="eyebrow">Pebble Intelligence</p>
            <h1>Trusted multilingual briefings</h1>
            <p>
              Upload a question in any language. Pebble will detect the source language, normalise the query, consult the
              RAG knowledge base, and deliver a structured reply back in the originating language.
            </p>
            <ul className="value-points">
              <li>Language detection & reversible translation</li>
              <li>FAISS-backed retrieval with FLAN-T5 reasoning</li>
              <li>Audit trail of every exchange</li>
            </ul>
          </div>
          <div className="pebble-image-wrapper">
            <img src="/image.avif" alt="Pebble smiling" loading="lazy" />
          </div>
        </section>

        <section className="chat-panel">
          <div className="chat-panel__messages" role="log" aria-live="polite" aria-label="Conversation with Pebble">
            {messages.map((message, index) => (
              <div
                key={`${message.sender}-${index}-${message.text.slice(0, 6)}`}
                className={`message-bubble ${message.sender === "user" ? "message-bubble--user" : "message-bubble--dino"}`}
              >
                <span className="message-bubble__label">{message.sender === "user" ? "You" : "Pebble"}</span>
                <p>{message.text}</p>
              </div>
            ))}
          </div>
          {activeLanguage && (
            <div className="language-pill" aria-live="polite">
              Working language: <span>{activeLanguage}</span>
            </div>
          )}
          <ChatInput onSubmit={handleSend} disabled={isProcessing} />
        </section>
      </div>
    </div>
  );
}

async function translateText(text: string, sourceLang: string, targetLang: string): Promise<string> {
  if (sourceLang === targetLang) {
    return text;
  }

  const response = await fetch(`${API_BASE_URL}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      source_lang: sourceLang,
      target_lang: targetLang,
    }),
  });

  if (!response.ok) {
    throw new Error("Translation failed");
  }

  const data: TranslationResponse = await response.json();
  return data.translated_text;
}

async function detectLanguageFromAPI(text: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/detect-language`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error("Failed to detect language");
  }

  const data: LanguageDetectionResponse = await response.json();
  return data.detected_language;
}
