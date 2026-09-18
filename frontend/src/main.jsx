import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { API_ORIGIN } from "./api";

// The API lives on another origin, so warm up DNS and TLS to it while React is
// still mounting. Without this the first request pays a full connection setup
// on top of its own round trip. The URL is an env var, hence the runtime tag.
if (API_ORIGIN && API_ORIGIN !== window.location.origin) {
  for (const rel of ["preconnect", "dns-prefetch"]) {
    const link = document.createElement("link");
    link.rel = rel;
    link.href = API_ORIGIN;
    if (rel === "preconnect") link.crossOrigin = "";
    document.head.appendChild(link);
  }
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
