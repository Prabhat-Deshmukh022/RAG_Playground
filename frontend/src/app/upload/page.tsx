'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast, Toaster } from 'sonner';

export default function UploadPage() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [ragOption, setRagOption] = useState<number>(1);
  const [llmOption, setLlmOption] = useState<number>(1);
  const router = useRouter()

    const goToQuery = () => {
        router.push(`query?option=${ragOption}`)
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

  return (
    <div className="max-w-xl mx-auto mt-12 p-6 rounded-xl border shadow-sm space-y-6">
      <h1 className="text-2xl font-semibold">Upload PDFs</h1>

      <div className="space-y-2">
        <Label htmlFor="files">Choose up to 3 PDF files</Label>
        <Input id="files" type="file" accept=".pdf" multiple onChange={handleFileChange} />
      </div>

      <div className="space-y-2">
        <Label htmlFor="rag_option">RAG Architecture</Label>
        <select
          id="rag_option"
          className="w-full border p-2 rounded-md"
          value={ragOption}
          onChange={(e) => setRagOption(Number(e.target.value))}
        >
          <option value={1}>Basic RAG</option>
          <option value={2}>Hybrid RAG</option>
          <option value={3}>Auto-Merge RAG</option>
        </select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="llm_option">LLM Option</Label>
        <select
          id="llm_option"
          className="w-full border p-2 rounded-md"
          value={llmOption}
          onChange={(e) => setLlmOption(Number(e.target.value))}
        >
          <option value={1}>Gemini</option>
          <option value={2}>Mistral</option>
          <option value={3}>Groq</option>
        </select>
      </div>

      <Button onClick={handleUpload}>Upload</Button>
        <br />
      <Button onClick={goToQuery}>Go to Query</Button>

      <Toaster />
    </div>
  );
}
