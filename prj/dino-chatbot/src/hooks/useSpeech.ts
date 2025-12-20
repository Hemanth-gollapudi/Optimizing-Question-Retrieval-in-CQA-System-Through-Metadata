// src/hooks/useSpeech.ts
import { useSpeechSynthesis } from "react-speech-kit";

export const useSpeech = () => {
  const { speak, voices } = useSpeechSynthesis();

  const speakText = (text: string) => {
    speak({ text, voice: voices[0] });
  };

  return { speakText };
};
