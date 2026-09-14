export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}
export async function apiResponse(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(options.headers);
  const token = sessionStorage.getItem("sew-session");
  if (token) headers.set("X-Sew-Session", token);
  if (path === "/auth/login") headers.set("X-Sew-Session-Transport", "header");
  headers.set("Accept", "application/json");
  if (options.body && typeof options.body === "string")
    headers.set("Content-Type", "application/json");
  const response = await fetch("/api" + path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  if ((response.status === 401 || (response.ok && path === "/auth/logout")) && sessionStorage.getItem("sew-session") === token) sessionStorage.removeItem("sew-session");
  const issued = response.headers.get("X-Sew-Session");
  if (response.ok && path === "/auth/login" && issued) sessionStorage.setItem("sew-session", issued);
  if (!response.ok) {
    const data = await response.json().catch(() => ({ error: "The server returned an unreadable response." }));
    throw new ApiError(data?.error || `Request failed (${response.status})`, response.status, data || {});
  }
  return response;
}
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await apiResponse(path, options);
  const data =
    response.status === 204
      ? null
      : await response
          .json()
          .catch(() => ({
            error: "The server returned an unreadable response.",
          }));
  return data as T;
}
export const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
});
export async function downloadFile(path: string, filename: string) {
  const response = await apiResponse(path.replace(/^\/api(?=\/)/, ""));
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
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
