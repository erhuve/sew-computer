import { useEffect, useState } from "react";
import { apiResponse } from "../lib/api";

export default function PrivateImage({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [image, setImage] = useState<{ source: string; url: string } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | undefined;
    apiResponse(src.replace(/^\/api(?=\/)/, ""), { signal: controller.signal })
      .then(response => response.blob())
      .then(blob => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setImage({ source: src, url: objectUrl });
      })
      .catch(() => { if (!controller.signal.aborted) setImage(null); });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);
  return <img src={image?.source === src ? image.url : undefined} alt={alt} className={className} />;
}
