import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import LandingScreen from "./screens/LandingScreen";
import { createBrowserRouter, RouterProvider } from "react-router";
import Session from "./screens/SessionScreen";
import "bootstrap/dist/css/bootstrap.min.css";
import SettingsScreen from "./screens/SettingsScreen";

const router = createBrowserRouter([
  { path: "/", element: <LandingScreen /> },
  { path: "/session", element: <Session /> },
  { path: "/settings", element: <SettingsScreen /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
