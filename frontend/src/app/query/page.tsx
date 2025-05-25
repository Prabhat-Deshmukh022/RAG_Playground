'use client';

import FormattedLLMResponse from "@/components/FormattedLLMResponse";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast, Toaster } from "sonner";

type ChatMessage = {
  sender: "user" | "bot";
  text: string;
};

export default function QueryPage() {
  const searchParams = useSearchParams();
  const optionFromQuery = searchParams.get("option") ?? "1";
  const [option] = useState<number>(parseInt(optionFromQuery, 10));
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);

  // No select for RAG option — fixed from URL param

  async function sendQuery() {
    if (!query.trim()) {
      toast.error("Please enter a query.");
      return;
    }
    setLoading(true);

    setMessages((msgs) => [...msgs, { sender: "user", text: query }]);

    try {
        const formData = new FormData()
        formData.append("query",query)
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query?option=${option}`, {
        method: "POST",
        // headers: {
        //   "Content-Type": "application/x-www-form-urlencoded",
        // },
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Something went wrong");
      }

      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: data.answer || "No answer received." },
      ]);
      setQuery("");
    } catch (error: any) {
      toast.error(error.message || "Failed to fetch answer");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto mt-12 p-6 bg-gray-900 rounded-xl shadow-lg text-white flex flex-col h-[80vh]">
      <h1 className="text-3xl font-semibold mb-4">RAG Chatbot</h1>

      <div className="mb-4">
        <p>
          <strong>RAG Architecture:</strong>{" "}
          {option === 1 ? "Basic RAG" : option === 2 ? "Hybrid RAG" : "Auto-Merge RAG"}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto mb-4 border border-gray-700 rounded-md p-4 bg-gray-800">
        {messages.length === 0 && (
          <p className="text-gray-400 text-center">No conversation yet. Ask something!</p>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`mb-3 max-w-[70%] rounded-lg px-4 py-2 ${
              msg.sender === "user"
                ? "bg-blue-600 self-end text-white"
                : "bg-gray-700 self-start text-gray-200"
            }`}
          >
            <FormattedLLMResponse text={msg.text}/>
            {/* {msg.text} */}
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <Input
          type="text"
          placeholder="Type your question here..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              sendQuery();
            }
          }}
        />
        <Button onClick={sendQuery} disabled={loading}>
          {loading ? "Loading..." : "Send"}
        </Button>
      </div>

      <Toaster />
    </div>
  );
}
