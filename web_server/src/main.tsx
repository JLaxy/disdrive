import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "bootstrap/dist/css/bootstrap.min.css";
import { createBrowserRouter, RouterProvider } from "react-router";
import StartScreen from "./screens/StartScreen.tsx";
import LogsScreen from "./screens/LogsScreen.tsx";
import SettingsScreen from "./screens/SettingsScreen.tsx";
import StartSessionScreen from "./screens/StartSessionScreen.tsx";

// Router
const reactRouter = createBrowserRouter([
  { path: "/", element: <StartScreen /> },
  { path: "/start_session", element: <StartSessionScreen /> },
  { path: "/logs", element: <LogsScreen /> },
  { path: "/settings", element: <SettingsScreen /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={reactRouter} />
  </StrictMode>
);
