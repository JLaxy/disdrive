import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import LandingScreen from "./screens/LandingScreen";
import { createBrowserRouter, RouterProvider } from "react-router";
import "bootstrap/dist/css/bootstrap.min.css";
import SettingsScreen from "./screens/SettingsScreen";
import LogsScreen from "./screens/LogsScreen";
import SessionScreen from "./screens/SessionScreen";

const router = createBrowserRouter([
  { path: "/", element: <LandingScreen /> },
  { path: "/session", element: <SessionScreen /> },
  { path: "/logs", element: <LogsScreen /> },
  { path: "/settings", element: <SettingsScreen /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
