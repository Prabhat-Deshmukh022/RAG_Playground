'use client';

import FormattedLLMResponse from "@/components/FormattedLLMResponse";
import { useState } from "react";
// import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select,SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast, Toaster } from "sonner";
import ContextToggleButton from "@/components/ContextToggleButton";

type ChatMessage = {
  sender: "user" | "bot";
  text: string;
  contexts?: string[];
};

export default function QueryPage() {
//   const searchParams = useSearchParams();
//   const optionFromQuery = searchParams.get("option") ?? "1";
//   const [option] = useState<number>(parseInt(optionFromQuery, 10));
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState<FileList | null>(null);
  const [ragOption, setRagOption] = useState<string>("1"); // default from URL
  const [llmOption, setLlmOption] = useState<string>("1"); // or "mistral" or any default

  // No select for RAG option — fixed from URL param

  async function sendQuery() {
    if (!query.trim()) {
      toast.error("Please enter a query.");
      return;
    }
    setLoading(true);

    setMessages((msgs) => 
        [
            ...msgs, 
            { 
                sender: "user", 
                text: query 
            }
        ]
    );

    try {
        const formData = new FormData()
        formData.append("query",query)
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query?option=${ragOption}`, {
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

      console.log(`Contexts ${data.contexts}`);

      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: data.answer || "No answer received.",contexts:data.contexts.slice(0,2) || [] },
      ]);
      setQuery("");
    } catch (error: unknown) {
      if (error instanceof Error) {
        toast.error(error.message || "Failed to fetch answer");
      } else {
        toast.error("Failed to fetch answer");
      }
    } finally {
      setLoading(false);
    }
  }

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 3) {
      toast.error('You can upload a maximum of 3 PDF files.');
      return;
    }
    setFiles(e.target.files);
  };

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      toast.error('Please select up to 3 PDF files.');
      return;
    }

    const formData = new FormData();
    formData.append('rag_option', ragOption.toString());
    formData.append('llm_option', llmOption.toString());
    Array.from(files).forEach((file) => formData.append('files', file));

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        toast.success('Files uploaded and ingested successfully.');
        console.log(data);
      } else {
        toast.error(data.detail || 'Something went wrong.');
      }
    } catch (err) {
      toast.error('Failed to connect to server.');
      console.error(err);
    }
  };

// ...existing code...
return (
  <div className="min-h-screen p-6 bg-gray-900 text-white flex flex-col">
    <div className="max-w-5xl mx-auto mt-12 p-6 bg-gray-900 rounded-xl shadow-lg text-white flex flex-col h-screen w-full">
      <h1 className="text-3xl font-semibold mb-4">RAG Chatbot</h1>

      <div className="mb-4">
        <p>
          <strong>RAG Architecture:</strong>{" "}
          {ragOption === "1" ? "Basic RAG" : ragOption === "2" ? "Hybrid RAG" : "Auto-Merge RAG"}
        </p>
      </div>

      {/* Chat/message area */}
      <div className="flex-1 flex flex-col overflow-y-auto mb-6 bg-gray-800 rounded-2xl shadow-lg p-6 w-full ">
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
            <FormattedLLMResponse text={msg.text} />
            {/* Only show context button for bot responses with context */}
            {msg.sender === "bot" && Array.isArray(msg.contexts) && msg.contexts.length > 0 && (
        <ContextToggleButton contexts={msg.contexts} />
        )}
          </div>
        ))}
      </div>

{/* Upload & Configuration */}
<div className="mb-6 p-4 border border-gray-700 rounded-md bg-gray-800 w-full">
  <div className="flex flex-col gap-4 w-full">
    <div className="flex flex-row gap-4 items-end">
      {/* File Upload */}
      <div className="flex-1 min-w-[180px]">
        <Label className="block mb-1">Upload</Label>
        <Input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="w-full"
        />
      </div>
      {/* RAG Option Select */}
      <div className="flex-1 min-w-[180px]">
        <Label className="block mb-1">RAG Architecture</Label>
        <Select value={ragOption} onValueChange={setRagOption}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select RAG option" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">Basic RAG</SelectItem>
            <SelectItem value="2">Hybrid RAG</SelectItem>
            <SelectItem value="3">Auto-Merge RAG</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {/* LLM Option Select */}
      <div className="flex-1 min-w-[180px]">
        <Label className="block mb-1">LLM</Label>
        <Select value={llmOption} onValueChange={setLlmOption}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select LLM" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">Gemini</SelectItem>
            <SelectItem value="2">Mistral</SelectItem>
            <SelectItem value="3">Groq</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {/* Upload Button */}
      <Button onClick={handleUpload} variant="secondary" className="ml-2 h-10 mt-6">
        Upload
      </Button>
    </div>
  </div>
</div>

      {/* Query input */}
      <div className="flex gap-2 items-center w-full">
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
  </div>
);
// ...existing code...

}
