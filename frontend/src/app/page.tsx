'use client';

import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import Image from "next/image";

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-10 bg-gray-50 dark:bg-gray-950 text-center">
      <Image
        src="/next.svg"
        alt="Next.js Logo"
        width={180}
        height={38}
        className="dark:invert mb-6"
        priority
      />
      <h1 className="text-4xl font-bold text-gray-800 dark:text-gray-100 mb-4">
        RAG Architecture Comparison
      </h1>
      <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-xl">
        Upload PDFs and explore answers from multiple Retrieval-Augmented Generation pipelines.
      </p>
      <Button onClick={() => router.push("/upload")}>
        Go to Upload Page
      </Button>
    </div>
  );
}
