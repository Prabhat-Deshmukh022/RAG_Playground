import { useState } from "react";
import { Button } from "./ui/button";

export default function ContextToggleButton({ contexts }: { contexts: string[] }) {
  const [show, setShow] = useState(false);

  return (
    <div className="mt-2">
      <Button
        variant="outline"
        className="text-xs px-2 py-1 h-auto"
        onClick={() => setShow(!show)}
      >
        {show ? "Hide Context" : "Show Context"}
      </Button>

      {show && (
        <div className="mt-2 text-sm bg-gray-600 p-3 rounded-lg space-y-2">
          {contexts.map((ctx, i) => (
            <div key={i} className="text-gray-200">
              <strong>Chunk {i + 1}:</strong> {ctx}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
