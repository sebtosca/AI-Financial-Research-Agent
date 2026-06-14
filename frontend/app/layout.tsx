import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Argent Research",
  description: "AI-powered financial research workspace"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
