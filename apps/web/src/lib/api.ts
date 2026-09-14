export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (options.body && typeof options.body === "string")
    headers.set("Content-Type", "application/json");
  const response = await fetch("/api" + path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  const data =
    response.status === 204
      ? null
      : await response
          .json()
          .catch(() => ({
            error: "The server returned an unreadable response.",
          }));
  if (!response.ok)
    throw new ApiError(
      data?.error || `Request failed (${response.status})`,
      response.status,
      data || {},
    );
  return data as T;
}
export const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
});
export function downloadJson(name: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
